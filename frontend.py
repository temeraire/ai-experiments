"""
Frontend module for Local LLM Logger v3
Generates HTML/React UI for the web interface
"""
from pathlib import Path

from config import DEFAULT_MODEL


def escape_html(s: str) -> str:
    """Escape HTML special characters"""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_index_html() -> str:
    """Generate the main HTML page with React UI"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Local LLM Logger v3</title>
<link rel="icon" type="image/png" href="/favicon"/>
<script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
<script crossorigin src="https://unpkg.com/@mui/material@5.15.14/umd/material-ui.development.js"></script>
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Inter:300,400,500,700&display=swap"/>
<style>
body{{margin:0;background:#fafafa}}
.container{{max-width:1200px;margin:32px auto;padding:0 16px}}
.chat-message{{margin:12px 0;padding:12px;border-radius:8px}}
.user-message{{background:#e3f2fd;text-align:right}}
.assistant-message{{background:#f5f5f5}}
.message-meta{{font-size:0.85em;color:#666;margin-top:4px}}
.summary-box{{background:#fff3cd;border-left:3px solid #ffc107;padding:8px 12px;margin-bottom:12px;border-radius:4px}}
.summary-box summary{{cursor:pointer;font-weight:600;color:#856404;font-size:0.9em;user-select:none}}
.summary-box summary:hover{{color:#533f03}}
.summary-content{{margin-top:8px;font-size:0.9em;color:#856404;line-height:1.5}}
@keyframes jump{{
  0%, 80%, 100% {{transform:translateY(0)}}
  40% {{transform:translateY(-10px)}}
}}
.throbber{{
  display:inline-flex;
  align-items:center;
  gap:4px;
  margin-right:8px;
}}
.throbber span{{
  display:inline-block;
  width:8px;
  height:8px;
  border-radius:50%;
  background:#2196f3;
  animation:jump 1s ease-in-out infinite;
}}
.throbber span:nth-child(2){{
  animation-delay:0.15s;
}}
.throbber span:nth-child(3){{
  animation-delay:0.3s;
}}
</style>
</head>
<body>
<div id="root"></div>
<script>
const e = React.createElement;
const {{useState, useEffect, useRef}} = React;
const {{Container, TextField, Button, Paper, Typography, Stack, Divider, Alert, Snackbar, Box, Chip, Card, CardContent, Select, MenuItem, FormControl, InputLabel, IconButton, Dialog, DialogTitle, DialogContent, DialogActions, List, ListItem, ListItemText}} = MaterialUI;

function App() {{
  const [sessionId, setSessionId] = useState(localStorage.getItem('session_id') || '');
  const [conversationId, setConversationId] = useState(null);
  const [model, setModel] = useState(localStorage.getItem('selected_model') || '{escape_html(DEFAULT_MODEL)}');
  const [availableModels, setAvailableModels] = useState(['{escape_html(DEFAULT_MODEL)}']);
  const [prompt, setPrompt] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [status, setStatus] = useState('');
  const [snack, setSnack] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationInfo, setConversationInfo] = useState(null);
  const [tokenStats, setTokenStats] = useState(null);
  const [conversationFiles, setConversationFiles] = useState([]);
  const [showFileDialog, setShowFileDialog] = useState(false);
  const [selectedFileContent, setSelectedFileContent] = useState(null);
  const [savedConversations, setSavedConversations] = useState([]);
  const [showLoadDialog, setShowLoadDialog] = useState(false);
  const [showSaveConfirmDialog, setShowSaveConfirmDialog] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const [selectedModels, setSelectedModels] = useState([]);
  const [compareResults, setCompareResults] = useState(null);
  const [bestModel, setBestModel] = useState(null);
  const [modelStatuses, setModelStatuses] = useState({{}});  // Track per-model status: pending/running/complete/error
  const [modelMetadata, setModelMetadata] = useState({{}});  // Track model pricing/paid status
  const [conversationCost, setConversationCost] = useState(0);  // Track cumulative cost in USD
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {{
    if (!sessionId) {{
      const newId = 'session_' + Date.now();
      localStorage.setItem('session_id', newId);
      setSessionId(newId);
    }}
    // Load available models
    fetch('/models/list').then(r=>r.json()).then(data=>{{
      if (data.models && data.models.length > 0) {{
        setAvailableModels(data.models);
        // Store model metadata for pricing/paid status
        if (data.modelsWithMetadata) {{
          const metaMap = {{}};
          data.modelsWithMetadata.forEach(m => {{ metaMap[m.name] = m; }});
          setModelMetadata(metaMap);
        }}
        const savedModel = localStorage.getItem('selected_model');
        // Only set to first model if no saved model and current model is default
        if (!savedModel && model === '{escape_html(DEFAULT_MODEL)}') {{
          const firstModel = data.models[0];
          setModel(firstModel);
          localStorage.setItem('selected_model', firstModel);
        }} else if (savedModel && data.models.includes(savedModel)) {{
          setModel(savedModel);
        }}
      }}
    }}).catch(err=>console.error('Failed to load models:', err));
  }}, []);

  // Save model selection to localStorage whenever it changes
  useEffect(() => {{
    if (model) {{
      localStorage.setItem('selected_model', model);
    }}
  }}, [model]);

  useEffect(() => {{
    chatEndRef.current?.scrollIntoView({{ behavior: 'smooth' }});
  }}, [chatHistory]);

  // Auto-save on page unload
  useEffect(() => {{
    const handleBeforeUnload = (e) => {{
      if (conversationId && chatHistory.length > 0) {{
        // Try to save the conversation before leaving
        fetch('/conversation/end', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{ conversation_id: conversationId }}),
          keepalive: true  // Important: ensures request completes even if page is closing
        }}).catch(() => {{
          // Ignore errors during unload
        }});
      }}
    }};

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }}, [conversationId, chatHistory]);

  // Periodic auto-save every 30 seconds
  useEffect(() => {{
    if (!conversationId || chatHistory.length === 0) return;

    const autoSaveInterval = setInterval(() => {{
      // Just ping the server to keep conversation alive
      // We don't actually "save" until end, but this ensures the conversation exists
      console.log('Auto-save check: conversation still active');
    }}, 30000);  // 30 seconds

    return () => clearInterval(autoSaveInterval);
  }}, [conversationId, chatHistory]);

  const handleNewConversation = () => {{
    // If there's an active conversation, ask to save it first
    if (conversationId && chatHistory.length > 0) {{
      setShowSaveConfirmDialog(true);
    }} else {{
      startNewConversation();
    }}
  }};

  const startNewConversation = async () => {{
    setShowSaveConfirmDialog(false);
    try {{
      const r = await fetch('/conversation/new', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{ session_id: sessionId }})
      }});
      const data = await r.json();
      setConversationId(data.conversation_id);
      setChatHistory([]);
      setConversationFiles([]);
      setConversationCost(0);  // Reset cost for new conversation
      setStatus('New conversation started');
      setSnack('New conversation started');
    }} catch (err) {{
      setSnack('Error starting conversation: ' + err);
    }}
  }};

  const saveAndStartNew = async () => {{
    await endConversation();
    await startNewConversation();
  }};

  const sendMessage = async () => {{
    if (!prompt.trim()) {{ setSnack('Please enter a message'); return; }}
    if (!conversationId) {{ await startNewConversation(); return; }}

    setIsLoading(true);
    setStatus('Generating response...');

    // Add user message to UI immediately
    const userMsg = {{ role: 'user', content: prompt, model }};
    setChatHistory(prev => [...prev, userMsg]);
    const currentPrompt = prompt;
    setPrompt('');

    try {{
      const r = await fetch('/conversation/send', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
          conversation_id: conversationId,
          model,
          prompt: currentPrompt
        }})
      }});

      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'HTTP ' + r.status);

      // Add assistant response
      setChatHistory(prev => [...prev, {{
        role: 'assistant',
        content: data.response,
        summary: data.summary || '',
        model: data.model,
        turn: data.turn_number,
        response_time: data.response_time
      }}]);

      setStatus(`Turn ${{data.turn_number}} • ${{data.response_time.toFixed(2)}}s`);
      setConversationInfo(data.conversation_info);
      setTokenStats(data.token_stats);
      if (data.cost !== undefined) {{
        setConversationCost(data.cost);
      }}
    }} catch (err) {{
      setSnack('Error: ' + err);
      setStatus('Error');
    }} finally {{
      setIsLoading(false);
    }}
  }};

  const endConversation = async () => {{
    if (!conversationId) return;
    try {{
      await fetch('/conversation/end', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{ conversation_id: conversationId }})
      }});
      setSnack('Conversation ended and saved');
      setConversationId(null);
      setChatHistory([]);
      setStatus('');
    }} catch (err) {{
      setSnack('Error ending conversation: ' + err);
    }}
  }};

  const clearContext = async () => {{
    if (!conversationId) return;
    try {{
      const response = await fetch('/conversation/clear-context', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{ conversation_id: conversationId }})
      }});
      const data = await response.json();
      setSnack('Context cleared - history preserved, token usage reduced');
      setTokenStats(data.token_stats);
    }} catch (err) {{
      setSnack('Error clearing context: ' + err);
    }}
  }};

  const compareModels = async () => {{
    if (!prompt.trim()) {{ setSnack('Please enter a message'); return; }}
    if (!conversationId) {{ await startNewConversation(); return; }}
    if (selectedModels.length < 2) {{ setSnack('Please select at least 2 models to compare'); return; }}

    console.log('Starting comparison with models:', selectedModels);
    setIsLoading(true);
    setStatus(`Comparing ${{selectedModels.length}} models...`);
    setBestModel(null);

    // Initialize model statuses
    const initialStatuses = {{}};
    selectedModels.forEach(m => {{ initialStatuses[m] = 'pending'; }});
    setModelStatuses(initialStatuses);

    // Initialize results structure
    const initialResults = {{}};
    selectedModels.forEach(m => {{
      initialResults[m] = {{ model: m, status: 'pending', response: null, response_time: null, error: null }};
    }});
    setCompareResults({{
      prompt: prompt,
      results: initialResults
    }});

    const currentPrompt = prompt;
    setPrompt('');

    try {{
      // Use fetch with streaming to handle SSE (EventSource doesn't support POST)
      const response = await fetch('/conversation/compare-stream', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
          conversation_id: conversationId,
          models: selectedModels,
          prompt: currentPrompt
        }})
      }});

      if (!response.ok) {{
        throw new Error('HTTP ' + response.status);
      }}

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {{
        const {{done, value}} = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, {{stream: true}});
        const lines = buffer.split('\\n\\n');
        buffer = lines.pop(); // Keep incomplete line in buffer

        for (const line of lines) {{
          if (!line.trim() || !line.startsWith('data: ')) continue;

          try {{
            const data = JSON.parse(line.substring(6)); // Remove 'data: ' prefix
            console.log('SSE event received:', data.type, data);

            if (data.type === 'init') {{
            // Initial event - models list confirmed
            const statuses = {{}};
            data.models.forEach(m => {{ statuses[m] = 'running'; }});
            setModelStatuses(statuses);
          }} else if (data.type === 'result') {{
            // Individual model completed
            setModelStatuses(prev => ({{
              ...prev,
              [data.model]: data.error ? 'error' : 'complete'
            }}));

            setCompareResults(prev => ({{
              ...prev,
              results: {{
                ...prev.results,
                [data.model]: {{
                  model: data.model,
                  status: data.error ? 'error' : 'complete',
                  response: data.response,
                  summary: data.summary || '',
                  response_time: data.response_time,
                  error: data.error
                }}
              }}
            }}));

            setStatus(`Completed ${{data.completed}}/${{data.total}} models...`);
            }} else if (data.type === 'complete') {{
              // All models finished
              setStatus(`Comparison complete • ${{selectedModels.length}} models`);
              setConversationInfo(data.conversation_info);
              setIsLoading(false);
            }}
          }} catch (parseErr) {{
            console.error('Error parsing SSE event:', parseErr, line);
          }}
        }}
      }}
    }} catch (err) {{
      setSnack('Error: ' + err);
      setStatus('Error');
      setIsLoading(false);
    }}
  }};

  const toggleModelSelection = (modelName) => {{
    setSelectedModels(prev =>
      prev.includes(modelName)
        ? prev.filter(m => m !== modelName)
        : [...prev, modelName]
    );
  }};

  const copyToClipboard = (text) => {{
    navigator.clipboard.writeText(text);
    setSnack('Copied to clipboard');
  }};

  const fetchSavedConversations = async () => {{
    try {{
      const r = await fetch('/conversations/list');
      const data = await r.json();
      setSavedConversations(data.conversations || []);
      setShowLoadDialog(true);
    }} catch (err) {{
      setSnack('Error loading conversations: ' + err);
    }}
  }};

  const loadSavedConversation = async (convId) => {{
    try {{
      const r = await fetch(`/conversations/load/${{convId}}`);
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Failed to load');

      // Restore conversation as active (allows continuing)
      const restoreR = await fetch('/conversation/restore', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
          conversation_id: data.conversation_id,
          session_id: sessionId
        }})
      }});
      const restoreData = await restoreR.json();

      if (!restoreR.ok) {{
        // If restore fails, still load as read-only
        console.warn('Failed to restore conversation, loading read-only');
      }}

      setConversationId(data.conversation_id);

      // Build chat history from loaded turns
      const history = [];
      data.turns.forEach(turn => {{
        history.push({{
          role: 'user',
          content: turn.prompt,
          turn: turn.turn_number
        }});
        history.push({{
          role: 'assistant',
          content: turn.response,
          model: turn.model,
          turn: turn.turn_number,
          response_time: turn.response_time
        }});
      }});

      setChatHistory(history);
      setStatus(`Loaded: ${{data.total_turns}} turns`);
      setShowLoadDialog(false);
      setSnack(`Conversation loaded - you can continue it`);
    }} catch (err) {{
      setSnack('Error loading conversation: ' + err);
    }}
  }};

  const handleKeyPress = (ev) => {{
    if (ev.key === 'Enter' && !ev.shiftKey) {{
      ev.preventDefault();
      sendMessage();
    }}
  }};

  const handleFileUpload = async (ev) => {{
    const file = ev.target.files[0];
    if (!file) return;

    if (!conversationId) {{
      setSnack('Please start a conversation first');
      return;
    }}

    const formData = new FormData();
    formData.append('file', file);
    formData.append('conversation_id', conversationId);

    try {{
      const r = await fetch('/upload', {{method: 'POST', body: formData}});
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'Upload failed');

      // Add file to conversation files list (NOT to prompt)
      setConversationFiles(prev => [...prev, {{name: file.name, size: data.size}}]);
      setSnack(`Uploaded: ${{file.name}}`);
    }} catch (err) {{
      setSnack('Upload error: ' + err);
    }}
    // Reset input
    ev.target.value = '';
  }};

  const viewFile = (fileName) => {{
    // For now, just show a message that files are attached to conversation
    setSnack(`File "${{fileName}}" is attached to this conversation`);
  }};

  const removeFile = (fileName) => {{
    setConversationFiles(prev => prev.filter(f => f.name !== fileName));
    setSnack(`Removed: ${{fileName}}`);
  }};

  const resendWithDifferentModel = async (originalPrompt, targetModel) => {{
    if (!conversationId) return;

    setIsLoading(true);
    setStatus(`Resending to ${{targetModel}}...`);

    try {{
      const r = await fetch('/conversation/send', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
          conversation_id: conversationId,
          model: targetModel,
          prompt: originalPrompt
        }})
      }});

      const data = await r.json();
      if (!r.ok) throw new Error(data.error || 'HTTP ' + r.status);

      // Add assistant response
      setChatHistory(prev => [...prev, {{
        role: 'assistant',
        content: data.response,
        summary: data.summary || '',
        model: data.model,
        turn: data.turn_number,
        response_time: data.response_time
      }}]);

      setStatus(`Turn ${{data.turn_number}} • ${{data.response_time.toFixed(2)}}s`);
      setConversationInfo(data.conversation_info);
      setTokenStats(data.token_stats);
      setSnack(`Response from ${{targetModel}} received`);
    }} catch (err) {{
      setSnack('Error: ' + err);
      setStatus('Error');
    }} finally {{
      setIsLoading(false);
    }}
  }};

  return e('div', {{className: 'container'}}, [
    e(Paper, {{elevation: 1, sx: {{p: 3, borderRadius: 4, mb: 2}}}}, [
      e(Stack, {{direction: 'row', justifyContent: 'space-between', alignItems: 'center', mb: 2}}, [
        e(Typography, {{variant: 'h4'}}, 'Local LLM Logger v3'),
        e(Stack, {{direction: 'row', spacing: 1}}, [
          conversationId && e(Chip, {{
            label: `Turn: ${{chatHistory.filter(m => m.role === 'assistant').length}}`,
            color: 'primary',
            size: 'small'
          }}),
          conversationId && e(Chip, {{
            label: model,
            color: 'secondary',
            size: 'small'
          }}),
          tokenStats && tokenStats.total_tokens > 0 && e(Chip, {{
            label: `Tokens: ${{(tokenStats.total_tokens / 1000).toFixed(1)}}k`,
            color: 'info',
            size: 'small',
            variant: 'outlined'
          }}),
          conversationCost > 0 && e(Chip, {{
            label: `Cost: $${{conversationCost.toFixed(4)}}`,
            color: 'success',
            size: 'small',
            variant: 'outlined',
            title: 'Estimated API cost for this conversation'
          }}),
          conversationId && e(Chip, {{
            label: `Context: ${{Math.min(chatHistory.length, 20)}}/${{chatHistory.length}} msgs`,
            color: chatHistory.length > 20 ? 'warning' : 'success',
            size: 'small',
            variant: 'outlined',
            title: 'Context window: last 10 turns (20 messages) sent to model'
          }})
        ])
      ]),

      e(Stack, {{direction: 'row', spacing: 2, mb: 2, alignItems: 'center'}}, [
        !compareMode && e(FormControl, {{key: 'model-select', sx: {{minWidth: 300}}}}, [
          e(InputLabel, {{id: 'model-label'}}, 'Model'),
          e(Select, {{
            labelId: 'model-label',
            label: 'Model',
            value: model,
            onChange: (ev) => setModel(ev.target.value),
            size: 'small'
          }}, availableModels.map(m => {{
            const meta = modelMetadata[m] || {{}};
            const isPaid = meta.isPaid || false;
            // Light green for paid available, darker when selected
            const bgColor = isPaid ? (m === model ? '#2e7d32' : '#c8e6c9') : undefined;
            const textColor = isPaid && m === model ? '#fff' : undefined;
            return e(MenuItem, {{
              key: m,
              value: m,
              sx: {{
                backgroundColor: bgColor,
                color: textColor,
                '&:hover': {{ backgroundColor: isPaid ? '#a5d6a7' : undefined }},
                '&.Mui-selected': {{ backgroundColor: isPaid ? '#2e7d32' : undefined, color: isPaid ? '#fff' : undefined }},
                '&.Mui-selected:hover': {{ backgroundColor: isPaid ? '#1b5e20' : undefined }}
              }}
            }}, isPaid ? `$ ${{m}}` : m);
          }}))
        ]),
        compareMode && e(Box, {{key: 'compare-mode-label', sx: {{minWidth: 300, display: 'flex', alignItems: 'center'}}}}, [
          e(Typography, {{variant: 'subtitle1', color: 'secondary'}}, `Comparing ${{selectedModels.length}} models`)
        ]),
        e(Button, {{
          key: 'compare-btn',
          variant: compareMode ? 'contained' : 'outlined',
          onClick: () => {{
            setCompareMode(!compareMode);
            setCompareResults(null);
            if (!compareMode) {{
              // Entering compare mode - select first 2 models by default
              setSelectedModels(availableModels.slice(0, Math.min(2, availableModels.length)));
            }}
          }},
          disabled: isLoading,
          color: compareMode ? 'secondary' : 'primary'
        }}, compareMode ? 'Exit Compare Mode' : 'Compare Models'),
        e(Button, {{
          key: 'new-conv-btn',
          variant: conversationId ? 'outlined' : 'contained',
          onClick: handleNewConversation,
          disabled: isLoading
        }}, 'New Conversation'),
        e(Button, {{
          key: 'load-conv-btn',
          variant: 'outlined',
          onClick: fetchSavedConversations,
          disabled: isLoading
        }}, 'Load Conversation'),
        conversationId && e(Button, {{
          key: 'clear-context-btn',
          variant: 'outlined',
          color: 'warning',
          onClick: clearContext,
          disabled: isLoading,
          title: 'Clear context to reduce tokens (history preserved)'
        }}, 'Clear Context'),
        conversationId && e(Button, {{
          key: 'end-save-btn',
          variant: 'outlined',
          color: 'error',
          onClick: endConversation,
          disabled: isLoading
        }}, 'End & Save')
      ]),

      compareMode && e(Box, {{sx: {{mb: 2, p: 2, bgcolor: '#f5f5f5', borderRadius: 2}}}}, [
        e(Typography, {{variant: 'subtitle2', sx: {{mb: 1}}}}, 'Select models to compare (minimum 2):'),
        e(Stack, {{direction: 'row', spacing: 1, flexWrap: 'wrap', gap: 1}},
          availableModels.map(m => {{
            const meta = modelMetadata[m] || {{}};
            const isPaid = meta.isPaid || false;
            const isSelected = selectedModels.includes(m);
            // Green colors for paid models
            const chipColor = isPaid ? (isSelected ? 'success' : 'default') : (isSelected ? 'primary' : 'default');
            return e(Chip, {{
              key: m,
              label: isPaid ? `$ ${{m}}` : m,
              onClick: () => toggleModelSelection(m),
              color: chipColor,
              variant: isSelected ? 'filled' : 'outlined',
              size: 'small',
              sx: isPaid && !isSelected ? {{ borderColor: '#4caf50', color: '#2e7d32' }} : undefined
            }});
          }})
        )
      ]),

      status && e(Alert, {{
        severity: status.startsWith('Error') ? 'error' : 'info',
        sx: {{mb: 2}}
      }}, status)
    ]),

    conversationId && e(Paper, {{elevation: 1, sx: {{p: 2, mb: 2, maxHeight: '60vh', overflowY: 'auto', borderRadius: 4}}}}, [
      chatHistory.length === 0 && e(Typography, {{color: 'text.secondary', align: 'center'}}, 'Start chatting...'),
      ...chatHistory.map((msg, idx) =>
        e('div', {{
          key: idx,
          className: msg.role === 'user' ? 'chat-message user-message' : 'chat-message assistant-message'
        }}, [
          // Show collapsible summary for assistant messages
          msg.role === 'assistant' && msg.summary && e('details', {{className: 'summary-box'}}, [
            e('summary', {{}}, 'Summary'),
            e('div', {{className: 'summary-content'}}, msg.summary)
          ]),
          e(Typography, {{variant: 'body1', sx: {{whiteSpace: 'pre-wrap'}}}}, msg.content),
          e('div', {{className: 'message-meta'}}, [
            msg.role === 'assistant' && msg.turn && `Turn ${{msg.turn}} • `,
            msg.model && `${{msg.model}}`,
            msg.response_time && ` • ${{msg.response_time.toFixed(2)}}s`
          ]),
          msg.role === 'user' && e(Box, {{sx: {{mt: 1}}}},
            e(Select, {{
              size: 'small',
              displayEmpty: true,
              value: '',
              disabled: isLoading,
              onChange: (ev) => {{
                if (ev.target.value) {{
                  resendWithDifferentModel(msg.content, ev.target.value);
                  ev.target.value = '';
                }}
              }},
              sx: {{fontSize: '0.75rem', height: '24px'}}
            }}, [
              e(MenuItem, {{value: '', disabled: true}}, 'Resend to different model...'),
              ...availableModels.map(m => e(MenuItem, {{key: m, value: m}}, m))
            ])
          )
        ])
      ),
      e('div', {{ref: chatEndRef}})
    ]),

    conversationId && compareMode && compareResults && e(Paper, {{elevation: 1, sx: {{p: 2, mb: 2, borderRadius: 4}}}}, [
      e(Typography, {{variant: 'h6', sx: {{mb: 2}}}}, `Comparison Results`),
      e(Typography, {{variant: 'subtitle2', sx: {{mb: 2, color: '#666'}}}}, `Prompt: "${{compareResults.prompt.substring(0, 100)}}${{compareResults.prompt.length > 100 ? '...' : ''}}"`),
      e(Box, {{sx: {{display: 'grid', gridTemplateColumns: `repeat(${{Math.min(selectedModels.length, 3)}}, 1fr)`, gap: 2}}}},
        selectedModels.map((modelName, idx) => {{
          const result = compareResults.results[modelName] || {{}};
          const status = modelStatuses[modelName] || 'pending';
          const isRunning = status === 'running' || status === 'pending';
          const isComplete = status === 'complete';
          const isError = status === 'error';

          return e(Paper, {{key: idx, elevation: 3, sx: {{p: 2, bgcolor: bestModel === modelName ? '#e3f2fd' : 'white'}}}}, [
            e(Stack, {{direction: 'row', justifyContent: 'space-between', alignItems: 'center', mb: 1}}, [
              e(Typography, {{variant: 'subtitle1', fontWeight: 'bold'}}, modelName),
              isRunning && e('span', {{className: 'throbber'}}, [
                e('span'),
                e('span'),
                e('span')
              ]),
              isComplete && e(Typography, {{variant: 'caption', color: 'success.main'}},
                `${{result.response_time.toFixed(2)}}s`
              ),
              isError && e(Typography, {{variant: 'caption', color: 'error'}}, 'Error')
            ]),
            isRunning && e(Typography, {{color: 'text.secondary', variant: 'body2', fontStyle: 'italic'}}, 'Generating response...'),
            isError && e(Typography, {{color: 'error', variant: 'body2'}}, `Error: ${{result.error}}`),
            isComplete && result.response && e(Box, {{}}, [
              // Show summary if available
              result.summary && e('details', {{className: 'summary-box', style: {{marginBottom: '12px'}}}}, [
                e('summary', {{}}, 'Summary'),
                e('div', {{className: 'summary-content'}}, result.summary)
              ]),
              e(Typography, {{variant: 'body2', sx: {{whiteSpace: 'pre-wrap', maxHeight: '400px', overflowY: 'auto', mb: 2}}}}, result.response),
              e(Stack, {{direction: 'row', spacing: 1}}, [
                e(Button, {{
                  size: 'small',
                  variant: bestModel === modelName ? 'contained' : 'outlined',
                  onClick: () => setBestModel(modelName),
                  startIcon: bestModel === modelName ? '⭐' : '☆'
                }}, 'Best'),
                e(Button, {{
                  size: 'small',
                  variant: 'outlined',
                  onClick: () => copyToClipboard(result.response)
                }}, '📋 Copy')
              ])
            ])
          ]);
        }})
      )
    ]),

    conversationId && e(Paper, {{elevation: 1, sx: {{p: 2, borderRadius: 4}}}}, [
      conversationFiles.length > 0 && e(Box, {{sx: {{mb: 2, p: 1, bgcolor: '#f5f5f5', borderRadius: 2}}}}, [
        e(Typography, {{variant: 'caption', sx: {{display: 'block', mb: 1, color: '#666'}}}}, 'Attached files (available to all messages in this conversation):'),
        e(Box, {{sx: {{display: 'flex', flexWrap: 'wrap', gap: 1}}}},
          conversationFiles.map((f, i) => e(Chip, {{
            key: i,
            label: `${{f.name}} (${{(f.size/1024).toFixed(1)}}KB)`,
            size: 'small',
            onClick: () => viewFile(f.name),
            onDelete: () => removeFile(f.name),
            color: 'primary',
            variant: 'outlined'
          }}))
        )
      ]),
      isLoading && e(Box, {{sx: {{display: 'flex', alignItems: 'center', mb: 1, p: 1, bgcolor: '#e3f2fd', borderRadius: 2}}}}, [
        e('span', {{className: 'throbber'}}, [
          e('span'),
          e('span'),
          e('span')
        ]),
        e(Typography, {{variant: 'body2', color: 'primary'}}, status || 'Generating...')
      ]),
      e(Box, {{sx: {{position: 'relative'}}}}, [
        e(TextField, {{
          label: 'Your message',
          multiline: true,
          minRows: 3,
          maxRows: 8,
          fullWidth: true,
          value: prompt,
          onChange: (ev) => setPrompt(ev.target.value),
          onKeyPress: handleKeyPress,
          disabled: isLoading,
          placeholder: 'Type your message... (Shift+Enter for new line, or click + to upload file)'
        }}),
        e('input', {{
          ref: fileInputRef,
          type: 'file',
          style: {{display: 'none'}},
          onChange: handleFileUpload,
          accept: '.txt,.md,.py,.js,.json,.csv,.html,.css,.java,.cpp,.c,.h,.go,.rs,.sh,.yaml,.yml,.xml,.sql,.ipynb,.docx,.pdf'
        }}),
        e(IconButton, {{
          onClick: () => fileInputRef.current?.click(),
          disabled: isLoading,
          sx: {{position: 'absolute', right: 8, top: 8}},
          title: 'Upload file'
        }}, '+')
      ]),
      e(Stack, {{direction: 'row', spacing: 2, sx: {{mt: 2}}}}, [
        e(Button, {{
          variant: 'contained',
          onClick: compareMode ? compareModels : sendMessage,
          disabled: isLoading || !prompt.trim() || (compareMode && selectedModels.length < 2)
        }}, isLoading ? 'Generating...' : (compareMode ? `Compare (${{selectedModels.length}} models)` : 'Send')),
        e(Button, {{
          variant: 'outlined',
          onClick: () => {{ setPrompt(''); }},
          disabled: isLoading
        }}, 'Clear')
      ])
    ]),

    !conversationId && e(Box, {{sx: {{textAlign: 'center', mt: 4}}}}, [
      e(Typography, {{variant: 'h6', color: 'text.secondary'}}, 'Click "New Conversation" to start'),
      e(Typography, {{variant: 'body2', color: 'text.secondary', mt: 1}},
        'All conversations are automatically logged with full tracking')
    ]),

    // Load Conversation Dialog
    showLoadDialog && e(Dialog, {{
      open: showLoadDialog,
      onClose: () => setShowLoadDialog(false),
      maxWidth: 'md',
      fullWidth: true
    }}, [
      e(DialogTitle, {{}}, 'Load Saved Conversation'),
      e(DialogContent, {{}}, [
        savedConversations.length === 0 && e(Typography, {{color: 'text.secondary'}}, 'No saved conversations found'),
        savedConversations.length > 0 && e(List, {{}}, savedConversations.map(conv => e(ListItem, {{
          key: conv.id,
          button: true,
          onClick: () => loadSavedConversation(conv.id)
        }}, [
          e(ListItemText, {{
            primary: conv.first_prompt || 'No prompt',
            secondary: `${{conv.total_turns}} turns • Models: ${{conv.models_used.join(', ')}} • ${{new Date(conv.start_time).toLocaleString()}}`
          }})
        ])))
      ]),
      e(DialogActions, {{}}, [
        e(Button, {{onClick: () => setShowLoadDialog(false)}}, 'Cancel')
      ])
    ]),

    // Save Confirmation Dialog
    showSaveConfirmDialog && e(Dialog, {{
      open: showSaveConfirmDialog,
      onClose: () => setShowSaveConfirmDialog(false)
    }}, [
      e(DialogTitle, {{}}, 'Save Current Conversation?'),
      e(DialogContent, {{}}, [
        e(Typography, {{}}, 'Would you like to save the current conversation before starting a new one?')
      ]),
      e(DialogActions, {{}}, [
        e(Button, {{
          onClick: () => {{
            setShowSaveConfirmDialog(false);
            startNewConversation();
          }}
        }}, 'No'),
        e(Button, {{
          onClick: saveAndStartNew,
          variant: 'contained',
          color: 'primary'
        }}, 'Yes')
      ])
    ]),

    snack && e(Snackbar, {{
      open: true,
      autoHideDuration: 3000,
      onClose: () => setSnack(''),
      message: snack
    }})
  ]);
}}

ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(App));
</script>
</body>
</html>"""
