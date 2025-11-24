"""
Routes module for Local LLM Logger v3
Handles all Flask API endpoints
"""
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

from flask import Flask, request, jsonify, Response, send_file

from config import CONVERSATIONS_DIR, CONTEXT_WINDOW_SIZE
from models import Conversation
from llm_client import call_llm, get_ollama_models, get_claude_models, get_gemini_models
from storage import save_turn_artifacts, save_comparison_artifacts, export_conversation_to_markdown, export_conversation_to_docx
from frontend import generate_index_html

# Active conversations by session_id
ACTIVE_CONVERSATIONS: Dict[str, Conversation] = {}


def register_routes(app: Flask):
    """Register all Flask routes"""

    @app.get("/")
    def index() -> Response:
        """Serve the main UI"""
        html = generate_index_html()
        return Response(html, mimetype="text/html")

    @app.get("/favicon")
    def favicon():
        """Serve the favicon"""
        favicon_path = Path(__file__).parent / "images" / "favicon.png"
        return send_file(favicon_path, mimetype="image/png")

    # --------------------------
    # Conversation Endpoints
    # --------------------------

    @app.post("/conversation/new")
    def new_conversation():
        """Start a new conversation"""
        data = request.get_json(force=True)
        session_id = data.get("session_id")

        conv_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        conv = Conversation(conv_id)
        ACTIVE_CONVERSATIONS[session_id] = conv

        return jsonify({
            "conversation_id": conv_id,
            "start_time": conv.start_time.isoformat()
        })

    @app.post("/conversation/send")
    def send_message():
        """Send a message in an active conversation"""
        data = request.get_json(force=True)
        conv_id = data.get("conversation_id")
        model = data.get("model")
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "missing prompt"}), 400

        # Find conversation
        conv = None
        for sid, c in ACTIVE_CONVERSATIONS.items():
            if c.id == conv_id:
                conv = c
                break

        if not conv:
            return jsonify({"error": "conversation not found"}), 404

        # Build full prompt with file context
        files_context = conv.get_files_context()
        full_prompt = prompt
        if files_context:
            full_prompt = files_context + "\n" + prompt

        # Call LLM (routes to Ollama or Claude automatically)
        # Use windowed messages to reduce context size
        windowed_messages = conv.get_windowed_messages(CONTEXT_WINDOW_SIZE)
        start_time = time.time()
        try:
            result = call_llm(model, windowed_messages + [{"role": "user", "content": full_prompt}])
            response_time = time.time() - start_time
        except Exception as e:
            return jsonify({"error": f"LLM call failed: {e}"}), 500

        # Save artifacts (use original prompt without file context for display)
        turn_num = len(conv.turns) + 1
        paths = save_turn_artifacts(conv, turn_num, model, prompt, result)

        # Add turn to conversation (store original prompt)
        conv.add_turn(model, prompt, result, response_time, paths)

        return jsonify({
            "conversation_id": conv.id,
            "turn_number": turn_num,
            "model": model,
            "response": result,
            "response_time": response_time,
            "paths": paths,
            "conversation_info": {
                "total_turns": len(conv.turns),
                "duration": (datetime.now() - conv.start_time).total_seconds(),
                "models_used": list(conv.models_used)
            },
            "token_stats": conv.get_token_stats()
        })

    @app.post("/conversation/compare")
    def compare_models():
        """Send prompt to multiple models in parallel for comparison"""
        data = request.get_json(force=True)
        conv_id = data.get("conversation_id")
        models = data.get("models", [])  # List of model names
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "missing prompt"}), 400

        if not models or len(models) == 0:
            return jsonify({"error": "no models specified"}), 400

        # Find conversation
        conv = None
        for sid, c in ACTIVE_CONVERSATIONS.items():
            if c.id == conv_id:
                conv = c
                break

        if not conv:
            return jsonify({"error": "conversation not found"}), 404

        # Build full prompt with file context
        files_context = conv.get_files_context()
        full_prompt = prompt
        if files_context:
            full_prompt = files_context + "\n" + prompt

        # Get windowed messages (same context for all models)
        windowed_messages = conv.get_windowed_messages(CONTEXT_WINDOW_SIZE)
        messages = windowed_messages + [{"role": "user", "content": full_prompt}]

        # Call all models in parallel
        import concurrent.futures
        results = []

        def call_model(model_name):
            start_time = time.time()
            try:
                result = call_llm(model_name, messages)
                response_time = time.time() - start_time
                return {
                    "model": model_name,
                    "response": result,
                    "response_time": response_time,
                    "error": None
                }
            except Exception as e:
                response_time = time.time() - start_time
                return {
                    "model": model_name,
                    "response": None,
                    "response_time": response_time,
                    "error": str(e)
                }

        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = [executor.submit(call_model, model) for model in models]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Sort results by original model order
        results_dict = {r["model"]: r for r in results}
        ordered_results = [results_dict[model] for model in models if model in results_dict]

        # Save comparison results to conversation history
        turn_num = len(conv.turns) + 1
        paths = save_comparison_artifacts(conv, turn_num, prompt, ordered_results)
        conv.add_comparison_turn(prompt, ordered_results, paths)

        return jsonify({
            "conversation_id": conv.id,
            "prompt": prompt,
            "results": ordered_results,
            "conversation_info": {
                "total_turns": len(conv.turns),
                "duration": (datetime.now() - conv.start_time).total_seconds(),
                "models_used": list(conv.models_used)
            }
        })

    @app.post("/conversation/compare-stream")
    def compare_models_stream():
        """Send prompt to multiple models in parallel with real-time SSE streaming"""
        data = request.get_json(force=True)
        conv_id = data.get("conversation_id")
        models = data.get("models", [])
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "missing prompt"}), 400

        if not models or len(models) == 0:
            return jsonify({"error": "no models specified"}), 400

        # Find conversation
        conv = None
        for sid, c in ACTIVE_CONVERSATIONS.items():
            if c.id == conv_id:
                conv = c
                break

        if not conv:
            return jsonify({"error": "conversation not found"}), 404

        # Build full prompt with file context
        files_context = conv.get_files_context()
        full_prompt = prompt
        if files_context:
            full_prompt = files_context + "\n" + prompt

        # Get windowed messages (same context for all models)
        windowed_messages = conv.get_windowed_messages(CONTEXT_WINDOW_SIZE)
        messages = windowed_messages + [{"role": "user", "content": full_prompt}]

        def generate_sse():
            """Generator function for Server-Sent Events"""
            import concurrent.futures
            results_collected = []

            def call_model_with_stream(model_name):
                """Call model and return result for SSE streaming"""
                start_time = time.time()
                try:
                    result = call_llm(model_name, messages)
                    response_time = time.time() - start_time
                    return {
                        "model": model_name,
                        "response": result,
                        "response_time": response_time,
                        "error": None
                    }
                except Exception as e:
                    response_time = time.time() - start_time
                    return {
                        "model": model_name,
                        "response": None,
                        "response_time": response_time,
                        "error": str(e)
                    }

            # Send initial event with list of models
            yield f"data: {json.dumps({'type': 'init', 'models': models, 'conversation_id': conv.id})}\n\n"

            # Use ThreadPoolExecutor for parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
                # Submit all tasks
                futures = {executor.submit(call_model_with_stream, model): model for model in models}

                # Stream results as they complete
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results_collected.append(result)

                    # Send SSE event for this model's completion
                    event_data = {
                        "type": "result",
                        "model": result["model"],
                        "response": result["response"],
                        "response_time": result["response_time"],
                        "error": result["error"],
                        "completed": len(results_collected),
                        "total": len(models)
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"

            # Sort results by original model order
            results_dict = {r["model"]: r for r in results_collected}
            ordered_results = [results_dict[model] for model in models if model in results_dict]

            # Save comparison results to conversation history
            turn_num = len(conv.turns) + 1
            paths = save_comparison_artifacts(conv, turn_num, prompt, ordered_results)
            conv.add_comparison_turn(prompt, ordered_results, paths)

            # Send final completion event
            final_data = {
                "type": "complete",
                "conversation_id": conv.id,
                "turn_number": turn_num,
                "results": ordered_results,
                "conversation_info": {
                    "total_turns": len(conv.turns),
                    "duration": (datetime.now() - conv.start_time).total_seconds(),
                    "models_used": list(conv.models_used)
                }
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        return Response(generate_sse(), mimetype="text/event-stream")

    @app.post("/conversation/clear-context")
    def clear_context():
        """Clear conversation context (keeps history logged, reduces token usage)"""
        data = request.get_json(force=True)
        conv_id = data.get("conversation_id")

        # Find conversation
        conv = None
        for c in ACTIVE_CONVERSATIONS.values():
            if c.id == conv_id:
                conv = c
                break

        if not conv:
            return jsonify({"error": "conversation not found"}), 404

        conv.clear_context()

        return jsonify({
            "conversation_id": conv.id,
            "message": "Context cleared successfully",
            "total_turns_logged": len(conv.turns),
            "token_stats": conv.get_token_stats()
        })

    @app.post("/conversation/end")
    def end_conversation():
        """End and save a conversation"""
        data = request.get_json(force=True)
        conv_id = data.get("conversation_id")

        # Find and remove conversation
        conv = None
        session_to_remove = None
        for sid, c in ACTIVE_CONVERSATIONS.items():
            if c.id == conv_id:
                conv = c
                session_to_remove = sid
                break

        if not conv:
            return jsonify({"error": "conversation not found"}), 404

        conv.end_conversation()

        if session_to_remove:
            del ACTIVE_CONVERSATIONS[session_to_remove]

        return jsonify({
            "conversation_id": conv.id,
            "total_turns": len(conv.turns),
            "duration_seconds": (conv.end_time - conv.start_time).total_seconds(),
            "conversation_dir": str(conv.conv_dir)
        })

    @app.get("/conversations/list")
    def list_conversations():
        """List all saved conversations with metadata"""
        conversations = []

        # Scan conversations directory
        if not CONVERSATIONS_DIR.exists():
            return jsonify({"conversations": []})

        for conv_dir in sorted(CONVERSATIONS_DIR.iterdir(), reverse=True):
            if not conv_dir.is_dir() or conv_dir.name.startswith('.'):
                continue

            # Try to load conversation metadata
            summary_file = conv_dir / "conversation.json"
            if summary_file.exists():
                try:
                    with summary_file.open() as f:
                        summary = json.load(f)

                    # Extract first prompt as preview
                    first_prompt = ""
                    if summary.get("turns") and len(summary["turns"]) > 0:
                        first_prompt = summary["turns"][0].get("prompt", "")[:100]

                    conversations.append({
                        "id": conv_dir.name,
                        "start_time": summary.get("start_time"),
                        "end_time": summary.get("end_time"),
                        "total_turns": len(summary.get("turns", [])),
                        "models_used": summary.get("models_used", []),
                        "first_prompt": first_prompt
                    })
                except Exception:
                    # Skip conversations with corrupted metadata
                    continue

        return jsonify({"conversations": conversations})

    @app.get("/conversations/load/<conv_id>")
    def load_conversation(conv_id: str):
        """Load a saved conversation with all turns"""
        conv_dir = CONVERSATIONS_DIR / conv_id
        if not conv_dir.exists():
            return jsonify({"error": "conversation not found"}), 404

        summary_file = conv_dir / "conversation.json"
        if not summary_file.exists():
            return jsonify({"error": "conversation metadata not found"}), 404

        try:
            with summary_file.open() as f:
                summary = json.load(f)

            # Load all turns from JSONL files
            turns = []
            for turn_data in summary.get("turns", []):
                # Handle both "turn" and "turn_number" field names
                turn_num = turn_data.get("turn") or turn_data.get("turn_number")
                if not turn_num:
                    continue

                turn_dir_pattern = f"turn_{turn_num:03d}_*"

                # Find the turn directory
                turn_dirs = list(conv_dir.glob(turn_dir_pattern))
                if turn_dirs:
                    turn_dir = turn_dirs[0]
                    jsonl_file = turn_dir / "turn.jsonl"

                    if jsonl_file.exists():
                        with jsonl_file.open() as f:
                            turn_info = json.loads(f.readline())
                            turns.append({
                                "turn_number": turn_num,
                                "timestamp": turn_info.get("timestamp"),
                                "model": turn_info.get("model"),
                                "prompt": turn_info.get("prompt"),
                                "response": turn_info.get("result_markdown"),
                                "response_time": turn_data.get("response_time")
                            })

            return jsonify({
                "conversation_id": conv_id,
                "start_time": summary.get("start_time"),
                "end_time": summary.get("end_time"),
                "total_turns": len(turns),
                "models_used": summary.get("models_used", []),
                "turns": turns
            })

        except Exception as e:
            return jsonify({"error": f"Failed to load conversation: {e}"}), 500

    @app.post("/conversation/restore")
    def restore_conversation():
        """Restore a saved conversation to active state (allows continuing)"""
        data = request.get_json(force=True)
        conv_id = data.get("conversation_id")
        session_id = data.get("session_id")

        if not conv_id or not session_id:
            return jsonify({"error": "conversation_id and session_id required"}), 400

        # Load conversation data
        conv_dir = CONVERSATIONS_DIR / conv_id
        if not conv_dir.exists():
            return jsonify({"error": "conversation not found"}), 404

        summary_file = conv_dir / "conversation.json"
        if not summary_file.exists():
            return jsonify({"error": "conversation metadata not found"}), 404

        try:
            with summary_file.open() as f:
                summary = json.load(f)

            # Create a new Conversation object with the loaded data
            conv = Conversation(conv_id, summary.get("start_time"))
            conv.conv_dir = conv_dir
            conv.models_used = set(summary.get("models_used", []))

            # Restore message history
            for turn_data in summary.get("turns", []):
                # Handle both "turn" and "turn_number" field names
                turn_num = turn_data.get("turn") or turn_data.get("turn_number")
                if not turn_num:
                    continue

                turn_dir_pattern = f"turn_{turn_num:03d}_*"
                turn_dirs = list(conv_dir.glob(turn_dir_pattern))

                if turn_dirs:
                    turn_dir = turn_dirs[0]
                    jsonl_file = turn_dir / "turn.jsonl"

                    if jsonl_file.exists():
                        with jsonl_file.open() as f:
                            turn_info = json.loads(f.readline())

                            # Add messages to conversation context
                            conv.messages.append({
                                "role": "user",
                                "content": turn_info.get("prompt")
                            })
                            conv.messages.append({
                                "role": "assistant",
                                "content": turn_info.get("result_markdown")
                            })

                            # Restore turns list
                            conv.turns.append({
                                "turn_number": turn_num,
                                "model": turn_info.get("model"),
                                "prompt": turn_info.get("prompt"),
                                "response": turn_info.get("result_markdown"),
                                "response_time": turn_data.get("response_time"),
                                "timestamp": turn_info.get("timestamp"),
                                "paths": {}
                            })

            # Add to active conversations
            ACTIVE_CONVERSATIONS[session_id] = conv

            return jsonify({
                "conversation_id": conv_id,
                "message": "Conversation restored successfully",
                "total_turns": len(conv.turns)
            })

        except Exception as e:
            return jsonify({"error": f"Failed to restore conversation: {e}"}), 500

    @app.get("/conversation/export/<conv_id>")
    def export_conversation(conv_id: str):
        """Export a specific conversation"""
        fmt = request.args.get("fmt", "json").lower()

        conv_dir = CONVERSATIONS_DIR / conv_id
        if not conv_dir.exists():
            return jsonify({"error": "conversation not found"}), 404

        summary_file = conv_dir / "conversation.json"
        if not summary_file.exists():
            return jsonify({"error": "conversation summary not found"}), 404

        if fmt == "json":
            return send_file(summary_file, as_attachment=True)

        # For other formats, compile all turns
        with summary_file.open() as f:
            summary = json.load(f)

        if fmt == "md":
            md_file = export_conversation_to_markdown(conv_id, summary, conv_dir)
            return send_file(md_file, as_attachment=True)

        if fmt == "docx":
            try:
                docx_file = export_conversation_to_docx(conv_id, summary, conv_dir)
                return send_file(docx_file, as_attachment=True)
            except Exception as e:
                return jsonify({"error": f"pandoc conversion failed: {e}"}), 500

        return jsonify({"error": "unsupported format; use json, md, or docx"}), 400

    @app.get("/models/list")
    def list_models():
        """List available models (Ollama + Claude + Gemini)"""
        ollama_models = get_ollama_models()
        claude_models = get_claude_models()
        gemini_models = get_gemini_models()

        # Combine models: Claude first, then Gemini, then Ollama
        all_models = claude_models + gemini_models + ollama_models

        return jsonify({"models": all_models})

    @app.post("/upload")
    def upload_file():
        """Handle file upload and add to conversation context"""
        if 'file' not in request.files:
            return jsonify({"error": "no file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "no file selected"}), 400

        # Get conversation_id from form data
        conv_id = request.form.get('conversation_id')
        if not conv_id:
            return jsonify({"error": "conversation_id required"}), 400

        # Find conversation
        conv = None
        for sid, c in ACTIVE_CONVERSATIONS.items():
            if c.id == conv_id:
                conv = c
                break

        if not conv:
            return jsonify({"error": "conversation not found"}), 404

        try:
            filename = file.filename.lower()
            content = file.read()

            # Handle different file types
            if filename.endswith('.pdf'):
                # Extract text from PDF
                try:
                    import PyPDF2
                    from io import BytesIO
                    pdf_reader = PyPDF2.PdfReader(BytesIO(content))
                    text_content = ""
                    for page_num, page in enumerate(pdf_reader.pages):
                        text_content += f"\n--- Page {page_num + 1} ---\n"
                        text_content += page.extract_text()
                    if not text_content.strip():
                        return jsonify({"error": "PDF appears to be empty or contains only images"}), 400
                except ImportError:
                    return jsonify({"error": "PyPDF2 not installed. Run: pip install PyPDF2"}), 500
                except Exception as e:
                    return jsonify({"error": f"Failed to read PDF: {e}"}), 400

            elif filename.endswith('.docx'):
                # Extract text from DOCX
                try:
                    import docx
                    from io import BytesIO
                    doc = docx.Document(BytesIO(content))
                    text_content = ""
                    for para in doc.paragraphs:
                        text_content += para.text + "\n"
                    if not text_content.strip():
                        return jsonify({"error": "DOCX appears to be empty"}), 400
                except ImportError:
                    return jsonify({"error": "python-docx not installed. Run: pip install python-docx"}), 500
                except Exception as e:
                    return jsonify({"error": f"Failed to read DOCX: {e}"}), 400

            elif filename.endswith('.ipynb'):
                # Parse Jupyter notebook
                try:
                    notebook = json.loads(content.decode('utf-8'))
                    text_content = f"# Jupyter Notebook: {file.filename}\n\n"

                    for i, cell in enumerate(notebook.get('cells', [])):
                        cell_type = cell.get('cell_type', 'unknown')
                        source = cell.get('source', [])

                        # Handle source as list or string
                        if isinstance(source, list):
                            source_text = ''.join(source)
                        else:
                            source_text = source

                        if cell_type == 'markdown':
                            text_content += f"\n## Markdown Cell {i+1}\n{source_text}\n"
                        elif cell_type == 'code':
                            text_content += f"\n## Code Cell {i+1}\n```python\n{source_text}\n```\n"
                            # Include outputs if available
                            outputs = cell.get('outputs', [])
                            if outputs:
                                text_content += "\n### Output:\n"
                                for output in outputs:
                                    if 'text' in output:
                                        out_text = output['text']
                                        if isinstance(out_text, list):
                                            out_text = ''.join(out_text)
                                        text_content += f"```\n{out_text}\n```\n"
                                    elif 'data' in output and 'text/plain' in output['data']:
                                        out_text = output['data']['text/plain']
                                        if isinstance(out_text, list):
                                            out_text = ''.join(out_text)
                                        text_content += f"```\n{out_text}\n```\n"

                    if not text_content.strip():
                        return jsonify({"error": "Notebook appears to be empty"}), 400
                except json.JSONDecodeError:
                    return jsonify({"error": "Invalid Jupyter notebook format"}), 400
                except Exception as e:
                    return jsonify({"error": f"Failed to read notebook: {e}"}), 400
            else:
                # Handle as regular text file
                try:
                    text_content = content.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        text_content = content.decode('latin-1')
                    except:
                        return jsonify({"error": "unable to read file as text"}), 400

            # Add file to conversation context
            conv.add_file(file.filename, text_content)

            return jsonify({
                "filename": file.filename,
                "size": len(content),
                "success": True
            })
        except Exception as e:
            return jsonify({"error": f"failed to read file: {e}"}), 500
