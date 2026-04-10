## Bug Fix: Compare Mode Not Running All Selected Models

### Issue
When in compare mode with 6 models selected, pressing Enter only runs one model (claude-sonnet-4-5-20250929) instead of comparing all 6 models.

### Root Cause
The `handleKeyPress` function in frontend.py (line 474-477) always calls `sendMessage()` when Enter is pressed, regardless of whether the user is in compare mode. It should check the `compareMode` state and call `compareModels()` instead when in compare mode.

**Current code:**
```javascript
const handleKeyPress = (ev) => {
  if (ev.key === 'Enter' && !ev.shiftKey) {
    ev.preventDefault();
    sendMessage();  // Always calls single model
  }
};
```

### Fix
Update `handleKeyPress` to check `compareMode` and call the appropriate function:
- If `compareMode === true`: call `compareModels()`
- If `compareMode === false`: call `sendMessage()`

### Implementation Plan
- [x] Read and understand the code
- [x] Identify the bug location (frontend.py:474-477)
- [x] Create plan in TODO.md
- [x] Commit and push current changes
- [x] Create new git branch for this bug fix (fix/compare-mode-enter-key)
- [x] Update `handleKeyPress` function to check compareMode
- [x] Flask app restarted successfully
- [ ] Manual testing required (user to test in browser)
- [x] Update TODO.md review section
- [ ] Create pull request

### Changes to Make
**File**: `frontend.py` (lines 474-477)

Change from:
```javascript
const handleKeyPress = (ev) => {
  if (ev.key === 'Enter' && !ev.shiftKey) {
    ev.preventDefault();
    sendMessage();
  }
};
```

To:
```javascript
const handleKeyPress = (ev) => {
  if (ev.key === 'Enter' && !ev.shiftKey) {
    ev.preventDefault();
    if (compareMode) {
      compareModels();
    } else {
      sendMessage();
    }
  }
};
```

### Testing
- [ ] Start app and enter compare mode
- [ ] Select 6 models
- [ ] Type a prompt and press Enter
- [ ] Verify all 6 models run in parallel
- [ ] Exit compare mode and verify single model still works

---

## REVIEW: Compare Mode Enter Key Fix (2025-12-05)

### Issue
When in compare mode with multiple models selected, pressing Enter would only run a single model (the default model) instead of running all selected models in parallel comparison.

### Root Cause
The `handleKeyPress` function in frontend.py did not check the `compareMode` state. It always called `sendMessage()` which runs a single model, even when the user was in compare mode and had selected multiple models.

### Changes Made
**File**: `frontend.py` (lines 474-483)

**Before**:
```javascript
const handleKeyPress = (ev) => {
  if (ev.key === 'Enter' && !ev.shiftKey) {
    ev.preventDefault();
    sendMessage();
  }
};
```

**After**:
```javascript
const handleKeyPress = (ev) => {
  if (ev.key === 'Enter' && !ev.shiftKey) {
    ev.preventDefault();
    if (compareMode) {
      compareModels();
    } else {
      sendMessage();
    }
  }
};
```

### Impact
- **Minimal code change**: Only modified the `handleKeyPress` function to add a conditional check
- **No features removed**: All existing functionality preserved
- **Better UX**: Compare mode now works as expected when pressing Enter
- **No breaking changes**: Single model mode continues to work as before

### Testing Required
User should test:
1. Enter compare mode and select 6 models
2. Type a prompt and press Enter
3. Verify all 6 models run in parallel (not just one)
4. Exit compare mode
5. Verify single model still works when pressing Enter

---

## Enhancement: Persistent Error Messages

### Issue
- Snackbar messages auto-hide after 3 seconds (`autoHideDuration: 3000`)
- Error messages disappear before user can read them
- User walked away and returned to just "error" with no details visible

### Plan
- [ ] Add `snackSeverity` state to track error vs info messages
- [ ] Modify setSnack calls to include severity
- [ ] Make Snackbar persist for errors (no auto-hide), auto-hide for success
- [ ] Add Alert component with close button for better error visibility

### Files to modify
- `frontend.py` - Snackbar component and setSnack calls

---

## Round: Orchestrator Re-skin + Music Tab

### Goal
Rebrand to "Orchestrator" with light editorial theme, section navigation, and a working Music tab (prompt-to-ABC score via abcjs). All existing features preserved.

### Plan
- [x] 1. Create git branch `feature/orchestrator-reskin`
- [x] 2. `frontend.py`: Wrap existing `App` in MUI `ThemeProvider` + `AppBar` ("Orchestrator" / "the boss") + permanent `Drawer` with sections: Conversation, Logbook, Music, Visuals
- [x] 3. Light editorial theme: EB Garamond display font, Inter body, warm off-white `#faf7f1`, ink primary, single accent
- [x] 4. Existing `App` function becomes `ConversationSection` — zero changes to its logic/controls
- [x] 5. Logbook + Visuals = "coming soon" placeholders
- [x] 6. Music section: prompt + key/meter/tempo/bars controls, abcjs CDN for score render + MIDI playback, download .abc
- [x] 7. `routes.py`: Add `POST /music/generate` — calls Ollama via `call_llm()` with ABC-only system prompt, extracts ABC
- [x] 8. Test: server boots, HTML verified to contain all existing features + new elements
- [ ] 9. Manual browser testing (user)

### What is NOT touched
- `config.py`, `models.py`, `llm_client.py`, `storage.py`, `main.py`
- All existing routes in `routes.py` (conversation/*, models/*, upload, export)
- Multi-model comparison, SSE streaming, file uploads, cost tracking, conversation browser

### Review

**Files changed:**
- `frontend.py` — added Orchestrator shell (ThemeProvider, AppBar, Drawer), renamed `App` to `ConversationSection`, added `MusicSection`, `Placeholder`, `Shell` components. Updated title/header styling to editorial theme. All existing conversation logic, compare mode, file uploads, cost tracking, SSE streaming unchanged.
- `routes.py` — added `POST /music/generate` route using existing `call_llm()`. No existing routes modified.

**Files NOT changed:** `config.py`, `models.py`, `llm_client.py`, `storage.py`, `main.py`

**Verification:**
- Python syntax: both files parse clean
- Server boots successfully on :5005
- HTML contains all 10 new markers (Orchestrator, Shell, ThemeProvider, etc.)
- HTML contains all 10 existing feature markers (compareModels, fileInputRef, tokenStats, etc.)
- JS brace count balanced: 460/460
