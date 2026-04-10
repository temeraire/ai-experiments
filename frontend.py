"""
Frontend module for Orchestrator (formerly Local LLM Logger v3)
Generates HTML/React UI for the web interface
"""
import json
from pathlib import Path

from config import DEFAULT_MODEL


def escape_html(s: str) -> str:
    """Escape HTML special characters"""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_index_html() -> str:
    """Generate the main HTML page with React UI"""
    default_model_js = json.dumps(DEFAULT_MODEL)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Orchestrator</title>
<link rel="icon" type="image/png" href="/favicon"/>
<script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
<script crossorigin src="https://unpkg.com/@mui/material@5.15.14/umd/material-ui.development.js"></script>
<script src="https://cdn.jsdelivr.net/npm/abcjs@6.2.3/dist/abcjs-basic-min.js"></script>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,700;1,400&family=Inter:wght@300;400;500;700&display=swap"/>
<style>
html,body{{margin:0;background:#faf7f1;color:#1a1a1a;font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif}}
.display{{font-family:'EB Garamond',Georgia,serif;letter-spacing:.2px}}
.ink-rule{{height:1px;background:#1a1a1a;opacity:.15;margin:24px 0}}
.abc-render svg{{max-width:100%;height:auto}}
.container{{max-width:1200px;margin:24px auto;padding:0 20px}}

/* Chat messages */
.chat-message{{margin:10px 0;padding:14px 18px;border-radius:12px;transition:box-shadow 0.3s ease}}
.user-message{{background:linear-gradient(135deg,#e3f2fd,#bbdefb);text-align:right;border:1px solid #90caf9}}
.assistant-message{{background:#ffffff;border:1px solid #e0e0e0;box-shadow:0 1px 3px rgba(0,0,0,0.06)}}
.assistant-message:hover{{box-shadow:0 2px 8px rgba(0,0,0,0.1)}}
.message-meta{{font-size:0.82em;color:#888;margin-top:6px}}

/* Summary boxes */
.summary-box{{background:linear-gradient(135deg,#fff8e1,#fff3cd);border-left:3px solid #ffc107;padding:10px 14px;margin-bottom:12px;border-radius:6px;transition:all 0.2s ease}}
.summary-box:hover{{box-shadow:0 1px 4px rgba(255,193,7,0.3)}}
.summary-box summary{{cursor:pointer;font-weight:600;color:#856404;font-size:0.9em;user-select:none}}
.summary-box summary:hover{{color:#533f03}}
.summary-content{{margin-top:8px;font-size:0.9em;color:#856404;line-height:1.5}}

/* Loading throbber */
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

/* Fade-in for new messages */
@keyframes fadeSlideIn{{
  from {{opacity:0;transform:translateY(12px)}}
  to {{opacity:1;transform:translateY(0)}}
}}
.chat-message{{animation:fadeSlideIn 0.4s ease-out}}

/* Completion pulse for comparison cards */
@keyframes completionPulse{{
  0% {{box-shadow:0 0 0 0 rgba(76,175,80,0.5)}}
  50% {{box-shadow:0 0 0 8px rgba(76,175,80,0.15)}}
  100% {{box-shadow:0 0 0 0 rgba(76,175,80,0)}}
}}
.card-complete{{
  animation:completionPulse 1.2s ease-out;
  border:2px solid #4caf50 !important;
  transition:border-color 0.3s ease;
}}

/* Completion banner */
@keyframes bannerSlideIn{{
  from {{opacity:0;transform:translateY(-8px)}}
  to {{opacity:1;transform:translateY(0)}}
}}
.completion-banner{{
  animation:bannerSlideIn 0.3s ease-out;
  border-radius:8px;
  padding:10px 16px;
  margin-bottom:12px;
  display:flex;
  align-items:center;
  gap:8px;
}}

/* Comparison card hover */
.compare-card{{
  transition:transform 0.2s ease, box-shadow 0.2s ease;
  border-radius:12px !important;
}}
.compare-card:hover{{
  transform:translateY(-2px);
  box-shadow:0 4px 16px rgba(0,0,0,0.12) !important;
}}
</style>
</head>
<body>
<div id="root"></div>
<script>
const e = React.createElement;
const {{useState, useEffect, useRef}} = React;
const M = MaterialUI;
const {{ThemeProvider, createTheme, CssBaseline, AppBar, Toolbar, Drawer, Container, TextField, Button, Paper, Typography, Stack, Divider, Alert, Snackbar, Box, Chip, Card, CardContent, Select, MenuItem, FormControl, InputLabel, IconButton, Dialog, DialogTitle, DialogContent, DialogActions, List, ListItem, ListItemButton, ListItemIcon, ListItemText}} = M;

const DEFAULT_MODEL = {default_model_js};

const theme = createTheme({{
  palette: {{
    mode: 'light',
    background: {{ default: '#faf7f1', paper: '#fffdf8' }},
    primary:   {{ main: '#1a1a1a' }},
    secondary: {{ main: '#8a4b1f' }},
    text: {{ primary: '#1a1a1a', secondary: '#5a5550' }},
    divider: 'rgba(26,26,26,0.12)'
  }},
  typography: {{
    fontFamily: "'Inter',-apple-system,BlinkMacSystemFont,sans-serif",
    h1: {{ fontFamily: "'EB Garamond',Georgia,serif", fontWeight: 500 }},
    h2: {{ fontFamily: "'EB Garamond',Georgia,serif", fontWeight: 500 }},
    h3: {{ fontFamily: "'EB Garamond',Georgia,serif", fontWeight: 500 }},
    h4: {{ fontFamily: "'EB Garamond',Georgia,serif", fontWeight: 500 }},
    h5: {{ fontFamily: "'EB Garamond',Georgia,serif", fontWeight: 500 }},
    h6: {{ fontFamily: "'EB Garamond',Georgia,serif", fontWeight: 500 }},
  }},
  shape: {{ borderRadius: 6 }},
  components: {{
    MuiButton: {{ styleOverrides: {{ root: {{ textTransform: 'none', borderRadius: 4 }} }} }},
    MuiPaper:  {{ styleOverrides: {{ root: {{ backgroundImage: 'none' }} }} }}
  }}
}});

const DRAWER_W = 220;
const SECTIONS = [
  {{ id: 'conversation', label: 'Conversation', glyph: '\\u00B6' }},
  {{ id: 'logbook',      label: 'Logbook',      glyph: '\\u00A7' }},
  {{ id: 'music',        label: 'Music',        glyph: '\\u266A' }},
  {{ id: 'visuals',      label: 'Visuals',      glyph: '\\u25C6' }},
];

function Shell({{ section, setSection, children }}) {{
  return e(Box, {{ sx: {{ display: 'flex', minHeight: '100vh' }} }}, [
    e(AppBar, {{ key:'ab', position:'fixed', elevation:0, color:'transparent',
                sx:{{ backdropFilter:'blur(6px)', backgroundColor:'rgba(250,247,241,0.85)',
                     borderBottom:'1px solid rgba(26,26,26,0.12)', zIndex:(t)=>t.zIndex.drawer+1 }} }},
      e(Toolbar, null, [
        e(Typography, {{ key:'t', variant:'h5', className:'display', sx:{{ flexGrow:1, letterSpacing:'.5px' }} }}, 'Orchestrator'),
        e(Typography, {{ key:'s', variant:'caption', sx:{{ fontStyle:'italic', color:'text.secondary' }} }}, 'the boss'),
      ])
    ),
    e(Drawer, {{ key:'dr', variant:'permanent',
                sx:{{ width:DRAWER_W, flexShrink:0,
                     '& .MuiDrawer-paper':{{ width:DRAWER_W, boxSizing:'border-box',
                       backgroundColor:'#f3ede0', borderRight:'1px solid rgba(26,26,26,0.12)' }} }} }}, [
      e(Toolbar, {{key:'sp'}}),
      e(List, {{key:'l'}},
        SECTIONS.map(s => e(ListItem, {{ key:s.id, disablePadding:true }},
          e(ListItemButton, {{ selected: section===s.id, onClick:()=>setSection(s.id) }}, [
            e(ListItemIcon, {{ key:'i', sx:{{ minWidth:32, fontFamily:"'EB Garamond',serif", fontSize:20 }} }}, s.glyph),
            e(ListItemText, {{ key:'t', primaryTypographyProps:{{ className:'display', fontSize:17 }}, primary:s.label }})
          ])
        ))
      )
    ]),
    e(Box, {{ key:'main', component:'main',
             sx:{{ flexGrow:1, p:3, mt:8, width:'100%' }} }}, children)
  ]);
}}

function Placeholder({{ title, subtitle, body }}) {{
  return e(Box, null, [
    e(Typography, {{ key:'t', variant:'h3', className:'display' }}, title),
    e(Typography, {{ key:'s', variant:'body2', sx:{{ fontStyle:'italic', color:'text.secondary', mt:.5 }} }}, subtitle),
    e('div', {{ key:'r', className:'ink-rule' }}),
    e(Paper, {{ key:'p', elevation:0, sx:{{p:4, border:'1px dashed rgba(26,26,26,0.25)', backgroundColor:'transparent'}} }},
      e(Typography, {{ variant:'body1', sx:{{ fontStyle:'italic', color:'text.secondary' }} }}, body)
    )
  ]);
}}

function MusicSection() {{
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [prompt, setPrompt] = useState('a melancholy waltz with a fiddle lead');
  const [key_, setKey] = useState('Dm');
  const [meter, setMeter] = useState('3/4');
  const [tempo, setTempo] = useState(96);
  const [bars, setBars] = useState(16);
  const [status, setStatus] = useState('');
  const [abc, setAbc] = useState('');
  const [paths, setPaths] = useState({{}});
  const [snack, setSnack] = useState('');
  const renderRef = useRef(null);
  const synthRef = useRef(null);
  const [playReady, setPlayReady] = useState(false);

  useEffect(() => {{
    if (!abc || !renderRef.current) return;
    try {{
      const visual = window.ABCJS.renderAbc(renderRef.current, abc, {{ responsive: 'resize' }});
      setPlayReady(false);
      if (window.ABCJS.synth && window.ABCJS.synth.supportsAudio()) {{
        const synth = new window.ABCJS.synth.CreateSynth();
        synth.init({{ visualObj: visual[0] }}).then(() =>
          synth.prime().then(() => {{ synthRef.current = synth; setPlayReady(true); }})
        ).catch(err => setSnack('Synth init failed: '+err));
      }}
    }} catch (err) {{
      setSnack('Score render failed: '+err);
    }}
  }}, [abc]);

  const generate = async () => {{
    setStatus('Composing...'); setAbc('');
    try {{
      const r = await fetch('/music/generate', {{
        method:'POST', headers:{{'Content-Type':'application/json'}},
        body: JSON.stringify({{ model, prompt, key: key_, meter, tempo: Number(tempo), bars: Number(bars) }})
      }});
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || ('HTTP '+r.status));
      setAbc(data.abc || '');
      setPaths(data.paths || {{}});
      setStatus('Done.');
    }} catch (err) {{
      setStatus('Error'); setSnack(String(err));
    }}
  }};

  const play = () => {{ if (synthRef.current) synthRef.current.start(); }};
  const stop = () => {{ if (synthRef.current) synthRef.current.stop(); }};
  const download = () => {{
    const blob = new Blob([abc], {{type:'text/plain'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'score.abc'; a.click();
    URL.revokeObjectURL(url);
  }};

  return e(Box, null, [
    e(Typography, {{ key:'t', variant:'h3', className:'display' }}, 'Music'),
    e(Typography, {{ key:'s', variant:'body2', sx:{{ fontStyle:'italic', color:'text.secondary', mt:.5 }} }}, 'the boss directs \\u2014 the model composes in ABC'),
    e('div', {{ key:'r', className:'ink-rule' }}),
    e(Paper, {{ key:'p', elevation:0, sx:{{p:3, border:'1px solid rgba(26,26,26,0.12)'}} }}, [
      e(Stack, {{key:'r1', direction:'row', spacing:2, sx:{{mb:2}}}}, [
        e(TextField, {{key:'m', label:'Model', fullWidth:true, value:model, onChange:(ev)=>setModel(ev.target.value)}}),
      ]),
      e(TextField, {{key:'pr', label:'Describe the piece', multiline:true, minRows:3, fullWidth:true, value:prompt, onChange:(ev)=>setPrompt(ev.target.value)}}),
      e(Stack, {{key:'r2', direction:'row', spacing:2, sx:{{mt:2}}}}, [
        e(FormControl, {{key:'k', sx:{{width:120}}}}, [
          e(InputLabel, {{key:'l', id:'kk'}}, 'Key'),
          e(Select, {{key:'s', labelId:'kk', label:'Key', value:key_, onChange:(ev)=>setKey(ev.target.value)}},
            ['C','G','D','A','E','F','Bb','Eb','Am','Em','Dm','Gm','Bm','F#m'].map(k => e(MenuItem,{{key:k,value:k}},k)))
        ]),
        e(FormControl, {{key:'me', sx:{{width:120}}}}, [
          e(InputLabel, {{key:'l', id:'mm'}}, 'Meter'),
          e(Select, {{key:'s', labelId:'mm', label:'Meter', value:meter, onChange:(ev)=>setMeter(ev.target.value)}},
            ['2/4','3/4','4/4','6/8','9/8','12/8'].map(m => e(MenuItem,{{key:m,value:m}},m)))
        ]),
        e(TextField, {{key:'t', label:'Tempo (bpm)', type:'number', value:tempo, onChange:(ev)=>setTempo(ev.target.value), sx:{{width:140}}}}),
        e(TextField, {{key:'b', label:'Bars', type:'number', value:bars, onChange:(ev)=>setBars(ev.target.value), sx:{{width:120}}}}),
      ]),
      e(Stack, {{key:'btns', direction:'row', spacing:2, sx:{{mt:2}}}}, [
        e(Button, {{key:'g', variant:'contained', onClick:generate}}, 'Compose'),
        e(Button, {{key:'p', variant:'outlined', onClick:play, disabled:!playReady}}, 'Play'),
        e(Button, {{key:'s', variant:'text', onClick:stop, disabled:!playReady}}, 'Stop'),
        e(Button, {{key:'d', variant:'text', onClick:download, disabled:!abc}}, 'Download .abc'),
      ]),
      status && e(Alert, {{key:'al', severity: status.startsWith('Error')?'error':'info', sx:{{mt:2}}}}, status),
      abc && e(Box, {{key:'score', sx:{{mt:3}}}}, [
        e(Divider, {{key:'d'}}),
        e(Typography, {{key:'h', variant:'h6', sx:{{mt:2, mb:1}}}}, 'Score'),
        e('div', {{ key:'r', ref: renderRef, className:'abc-render' }}),
        e(Typography, {{key:'h2', variant:'h6', sx:{{mt:2}}}}, 'ABC source'),
        e('pre', {{key:'pre', style:{{whiteSpace:'pre-wrap', background:'#f3ede0', padding:12, borderRadius:4}}}}, abc),
        paths.turn_dir && e(Typography, {{key:'pt', variant:'caption', sx:{{color:'text.secondary'}}}}, 'Saved to: '+paths.turn_dir),
      ]),
      snack && e(Snackbar, {{key:'sn', open:true, autoHideDuration:2500, onClose:()=>setSnack(''), message:snack}})
    ])
  ]);
}}

function ConversationSection() {{
  const [sessionId, setSessionId] = useState(localStorage.getItem('session_id') || '');
  const [conversationId, setConversationId] = useState(null);
  const [model, setModel] = useState(localStorage.getItem('selected_model') || '{escape_html(DEFAULT_MODEL)}');
  const [availableModels, setAvailableModels] = useState(['{escape_html(DEFAULT_MODEL)}']);
  const [prompt, setPrompt] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [status, setStatus] = useState('');
  const [snack, setSnack] = useState('');
  const [snackSeverity, setSnackSeverity] = useState('info');  // 'error' persists, 'info'/'success' auto-hide
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
  const [completionBanner, setCompletionBanner] = useState(null);  // {{model, time, type}} for completion alerts
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

  const showCompletionBanner = (bannerData) => {{
    setCompletionBanner(bannerData);
    setTimeout(() => setCompletionBanner(null), 4000);
  }};

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
      setSnackSeverity('error'); setSnack('Error starting conversation: ' + err);
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
      showCompletionBanner({{model: data.model, time: data.response_time, type: 'single'}});
    }} catch (err) {{
      setSnackSeverity('error'); setSnack('Error: ' + err);
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
      setSnackSeverity('error'); setSnack('Error ending conversation: ' + err);
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
      setSnackSeverity('error'); setSnack('Error clearing context: ' + err);
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
              showCompletionBanner({{type: 'comparison', count: selectedModels.length}});
            }}
          }} catch (parseErr) {{
            console.error('Error parsing SSE event:', parseErr, line);
          }}
        }}
      }}
    }} catch (err) {{
      setSnackSeverity('error'); setSnack('Error: ' + err);
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
      setSnackSeverity('error'); setSnack('Error loading conversations: ' + err);
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
      setSnackSeverity('error'); setSnack('Error loading conversation: ' + err);
    }}
  }};

  const handleKeyPress = (ev) => {{
    if (ev.key === 'Enter' && !ev.shiftKey) {{
      ev.preventDefault();
      if (compareMode) {{
        compareModels();
      }} else {{
        sendMessage();
      }}
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
      setSnackSeverity('error'); setSnack('Upload error: ' + err);
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
      setSnackSeverity('error'); setSnack('Error: ' + err);
      setStatus('Error');
    }} finally {{
      setIsLoading(false);
    }}
  }};

  return e('div', {{className: 'container'}}, [
    e(Paper, {{elevation: 0, sx: {{p: 3, borderRadius: 4, mb: 2, border: '1px solid rgba(26,26,26,0.12)'}}}}, [
      e(Stack, {{direction: 'row', justifyContent: 'space-between', alignItems: 'center', mb: 2}}, [
        e(Typography, {{variant: 'h4', className: 'display', sx: {{fontWeight: 500, letterSpacing: '.3px'}}}}, 'Conversation'),
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

    completionBanner && e('div', {{className: 'completion-banner', style: {{
      background: completionBanner.type === 'comparison'
        ? 'linear-gradient(135deg, #e8f5e9, #c8e6c9)'
        : 'linear-gradient(135deg, #e3f2fd, #bbdefb)',
      border: completionBanner.type === 'comparison' ? '1px solid #81c784' : '1px solid #64b5f6'
    }}}}, [
      e(Chip, {{
        label: completionBanner.type === 'comparison'
          ? `All ${{completionBanner.count}} models complete`
          : `${{completionBanner.model}} responded in ${{completionBanner.time.toFixed(2)}}s`,
        color: completionBanner.type === 'comparison' ? 'success' : 'primary',
        variant: 'filled',
        size: 'small',
        sx: {{fontWeight: 600}}
      }})
    ]),

    conversationId && e(Paper, {{elevation: 2, sx: {{p: 2, mb: 2, maxHeight: '60vh', overflowY: 'auto', borderRadius: 4, border: '1px solid #e8e8e8'}}}}, [
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

    conversationId && compareMode && compareResults && e(Paper, {{elevation: 2, sx: {{p: 3, mb: 2, borderRadius: 4, border: '1px solid #e0e0e0'}}}}, [
      e(Typography, {{variant: 'h6', sx: {{mb: 1, fontWeight: 600}}}}, 'Comparison Results'),
      e(Typography, {{variant: 'subtitle2', sx: {{mb: 2, color: '#888'}}}}, `Prompt: "${{compareResults.prompt.substring(0, 100)}}${{compareResults.prompt.length > 100 ? '...' : ''}}"`),
      e(Box, {{sx: {{display: 'grid', gridTemplateColumns: `repeat(${{Math.min(selectedModels.length, 3)}}, 1fr)`, gap: 2}}}},
        selectedModels.map((modelName, idx) => {{
          const result = compareResults.results[modelName] || {{}};
          const status = modelStatuses[modelName] || 'pending';
          const isRunning = status === 'running' || status === 'pending';
          const isComplete = status === 'complete';
          const isError = status === 'error';
          const meta = modelMetadata[modelName] || {{}};
          const isPaid = meta.isPaid || false;

          return e(Paper, {{key: idx, elevation: 3, className: `compare-card ${{isComplete ? 'card-complete' : ''}}`, sx: {{p: 0, bgcolor: bestModel === modelName ? '#e3f2fd' : 'white', overflow: 'hidden'}}}}, [
            e(Box, {{sx: {{p: 1.5, background: isPaid ? 'linear-gradient(135deg, #e8f5e9, #c8e6c9)' : 'linear-gradient(135deg, #e3f2fd, #bbdefb)', borderBottom: '1px solid', borderColor: isPaid ? '#a5d6a7' : '#90caf9', display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}}}, [
              e(Stack, {{direction: 'row', spacing: 1, alignItems: 'center'}}, [
                e(Typography, {{variant: 'subtitle2', fontWeight: 700}}, modelName),
                isPaid && e(Chip, {{label: '$', size: 'small', color: 'success', sx: {{height: 20, fontSize: '0.7rem'}}}}),
              ]),
              isRunning && e('span', {{className: 'throbber'}}, [
                e('span'),
                e('span'),
                e('span')
              ]),
              isComplete && e(Chip, {{label: `${{result.response_time.toFixed(2)}}s`, size: 'small', color: 'success', variant: 'outlined', sx: {{height: 22, fontWeight: 600}}}}),
              isError && e(Chip, {{label: 'Error', size: 'small', color: 'error', sx: {{height: 22}}}})
            ]),
            e(Box, {{sx: {{p: 2}}}}, [
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
                }}, 'Copy')
              ])
            ])
            ]) // close inner Box (card body)
          ]);
        }})
      )
    ]),

    conversationId && e(Paper, {{elevation: 2, sx: {{p: 2, borderRadius: 4, border: '1px solid #e0e0e0'}}}}, [
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
      autoHideDuration: snackSeverity === 'error' ? null : 3000,
      onClose: () => {{ setSnack(''); setSnackSeverity('info'); }}
    }}, e(Alert, {{
      onClose: () => {{ setSnack(''); setSnackSeverity('info'); }},
      severity: snackSeverity,
      sx: {{ width: '100%' }}
    }}, snack))
  ]);
}}

function App() {{
  const [section, setSection] = useState('conversation');

  let body;
  if (section === 'conversation') body = e(ConversationSection);
  else if (section === 'music')   body = e(MusicSection);
  else if (section === 'logbook') body = e(Placeholder, {{ title:'Logbook', subtitle:'every turn, archived', body:'Coming soon \\u2014 browse and export your daily sessions. The archive is already being written to ./conversations/ on every turn.' }});
  else                            body = e(Placeholder, {{ title:'Visuals', subtitle:'shapes for ideas', body:'Coming soon \\u2014 diagrams, charts, and other visual representations of model output.' }});

  return e(ThemeProvider, {{ theme }}, [
    e(CssBaseline, {{ key:'c' }}),
    e(Shell, {{ key:'s', section, setSection }}, body)
  ]);
}}

ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(App));
</script>
</body>
</html>"""
