"""
Data models for Local LLM Logger v3
Handles Conversation class and related data structures
"""
import csv
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from config import CONVERSATIONS_CSV, TURNS_CSV, CONVERSATIONS_DIR, HAS_CLAUDE


class Conversation:
    """Represents a conversation with turn tracking and context management"""

    def __init__(self, conv_id: str, start_time: Optional[str] = None):
        self.id = conv_id
        self.start_time = datetime.fromisoformat(start_time) if start_time else datetime.now()
        self.end_time: Optional[datetime] = None
        self.turns: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, str]] = []  # For chat API
        self.models_used: set = set()
        self.files: List[Dict[str, Any]] = []  # Store uploaded files for conversation context
        self.conv_dir = CONVERSATIONS_DIR / conv_id
        self.conv_dir.mkdir(parents=True, exist_ok=True)

        # Token tracking for Claude models
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.rate_limit_reset_time: Optional[datetime] = None

    def add_file(self, filename: str, content: str):
        """Add a file to the conversation context"""
        file_entry = {
            "filename": filename,
            "content": content,
            "uploaded_at": datetime.now().isoformat()
        }
        self.files.append(file_entry)

    def get_files_context(self) -> str:
        """Get formatted string of all files in conversation"""
        if not self.files:
            return ""

        context_parts = []
        for f in self.files:
            context_parts.append(f"\n--- File: {f['filename']} ---\n{f['content']}\n--- End of file ---\n")
        return "\n".join(context_parts)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a given text"""
        if not HAS_CLAUDE:
            # Rough estimation: ~4 characters per token
            return len(text) // 4

        try:
            # Use tiktoken for Claude models (cl100k_base encoding)
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback to rough estimation
            return len(text) // 4

    def get_token_stats(self) -> Dict[str, Any]:
        """Get current token usage statistics"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "rate_limit_reset_time": self.rate_limit_reset_time.isoformat() if self.rate_limit_reset_time else None
        }

    def get_windowed_messages(self, window_size: int = 0) -> List[Dict[str, str]]:
        """Get messages with sliding window (keep only recent N turns)"""
        if window_size <= 0:
            # Return all messages (unlimited context)
            return self.messages

        # Each turn = 2 messages (user + assistant)
        # So window_size turns = window_size * 2 messages
        max_messages = window_size * 2

        if len(self.messages) <= max_messages:
            return self.messages

        # Return only the most recent messages
        return self.messages[-max_messages:]

    def clear_context(self):
        """Clear conversation context (but keep all turns logged)"""
        # Clear messages array (context for LLM)
        # But keep turns array (logged history)
        self.messages = []

    def add_turn(self, model: str, prompt: str, response: str,
                 response_time: float, paths: Dict[str, str],
                 input_tokens: int = 0, output_tokens: int = 0):
        """Add a turn to the conversation"""
        turn_num = len(self.turns) + 1

        # Track tokens (estimate if not provided)
        if input_tokens == 0:
            input_tokens = self.estimate_tokens(prompt)
        if output_tokens == 0:
            output_tokens = self.estimate_tokens(response)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        turn = {
            "turn_number": turn_num,
            "timestamp": datetime.now(),
            "type": "regular",
            "model": model,
            "prompt": prompt,
            "response": response,
            "response_time": response_time,
            "paths": paths,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        self.turns.append(turn)
        self.models_used.add(model)

        # Add to messages for chat continuity
        self.messages.append({"role": "user", "content": prompt})
        self.messages.append({"role": "assistant", "content": response})

        # Log turn to CSV
        self._log_turn(turn)

    def add_comparison_turn(self, prompt: str, results: List[Dict[str, Any]], paths: Dict[str, str]):
        """Add a comparison turn to the conversation"""
        turn_num = len(self.turns) + 1

        # Calculate total tokens and track models from all results
        total_input_tokens = 0
        total_output_tokens = 0
        total_response_time = 0

        for result in results:
            if result.get('error') is None:
                # Estimate tokens for each model's response
                input_tokens = self.estimate_tokens(prompt)
                output_tokens = self.estimate_tokens(result['response'])
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_response_time += result['response_time']

                # Track models used
                self.models_used.add(result['model'])

        self.total_input_tokens += total_input_tokens
        self.total_output_tokens += total_output_tokens

        turn = {
            "turn_number": turn_num,
            "timestamp": datetime.now(),
            "type": "comparison",
            "prompt": prompt,
            "results": results,  # Store all model results
            "response_time": total_response_time,
            "paths": paths,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens
        }
        self.turns.append(turn)

        # Note: We don't add comparison results to messages array
        # since they're meant to be compared, not used as context

        # Log each result to CSV
        self._log_comparison_turn(turn)

    def _log_turn(self, turn: Dict[str, Any]):
        """Log a turn to CSV file"""
        with TURNS_CSV.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.id,
                turn["turn_number"],
                turn["timestamp"].isoformat(),
                turn["model"],
                turn["prompt"],
                turn["response"],
                f"{turn['response_time']:.2f}",
                turn["paths"].get("jsonl_path", ""),
                turn["paths"].get("md_path", ""),
                turn["paths"].get("html_path", ""),
                turn["paths"].get("docx_path", "")
            ])

    def _log_comparison_turn(self, turn: Dict[str, Any]):
        """Log a comparison turn to CSV file (one row per model result)"""
        with TURNS_CSV.open("a", newline="") as f:
            writer = csv.writer(f)
            for result in turn["results"]:
                # Log each model's result as a separate row with same turn number
                writer.writerow([
                    self.id,
                    turn["turn_number"],
                    turn["timestamp"].isoformat(),
                    result["model"],
                    turn["prompt"],
                    result.get("response", f"ERROR: {result.get('error', 'Unknown error')}"),
                    f"{result['response_time']:.2f}",
                    turn["paths"].get("jsonl_path", ""),
                    turn["paths"].get("md_path", ""),
                    turn["paths"].get("html_path", ""),
                    turn["paths"].get("docx_path", "")
                ])

    def end_conversation(self):
        """End the conversation and save summary"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        # Log conversation to CSV
        with CONVERSATIONS_CSV.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.id,
                self.start_time.isoformat(),
                self.end_time.isoformat(),
                f"{duration:.2f}",
                len(self.turns),
                ",".join(sorted(self.models_used)),
                str(self.conv_dir)
            ])

        # Save conversation summary
        self._save_summary()

    def _save_summary(self):
        """Save conversation summary to JSON file"""
        # Build turns list, handling both regular and comparison turns
        turns_summary = []
        for t in self.turns:
            if t.get("type") == "comparison":
                # Comparison turn - include all model results
                turns_summary.append({
                    "turn": t["turn_number"],
                    "timestamp": t["timestamp"].isoformat(),
                    "type": "comparison",
                    "prompt": t["prompt"],
                    "results": t["results"],
                    "response_time": t["response_time"]
                })
            else:
                # Regular turn - single model/response
                turns_summary.append({
                    "turn": t["turn_number"],
                    "timestamp": t["timestamp"].isoformat(),
                    "type": "regular",
                    "model": t["model"],
                    "prompt": t["prompt"],
                    "response": t["response"],
                    "response_time": t["response_time"]
                })

        summary = {
            "conversation_id": self.id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            "total_turns": len(self.turns),
            "models_used": list(self.models_used),
            "turns": turns_summary
        }

        summary_path = self.conv_dir / "conversation.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def to_dict(self):
        """Convert conversation to dictionary for API responses"""
        return {
            "id": self.id,
            "start_time": self.start_time.isoformat(),
            "total_turns": len(self.turns),
            "models_used": list(self.models_used),
            "turns": [
                {
                    "turn": t["turn_number"],
                    "timestamp": t["timestamp"].isoformat(),
                    "model": t["model"],
                    "prompt": t["prompt"][:100] + "..." if len(t["prompt"]) > 100 else t["prompt"],
                    "response": t["response"][:200] + "..." if len(t["response"]) > 200 else t["response"]
                }
                for t in self.turns
            ]
        }


def initialize_csv_files():
    """Initialize CSV log files if they don't exist"""
    # Initialize conversations CSV
    if not CONVERSATIONS_CSV.exists():
        with CONVERSATIONS_CSV.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "conversation_id", "start_time", "end_time", "duration_seconds",
                "total_turns", "models_used", "conversation_dir"
            ])

    # Initialize turns CSV
    if not TURNS_CSV.exists():
        with TURNS_CSV.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "conversation_id", "turn_number", "timestamp", "model",
                "prompt", "response", "response_time_seconds",
                "jsonl_path", "md_path", "html_path", "docx_path"
            ])
