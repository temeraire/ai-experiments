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
- [ ] Commit and push current changes
- [ ] Create new git branch for this bug fix
- [ ] Update `handleKeyPress` function to check compareMode
- [ ] Test the fix
- [ ] Update TODO.md review section
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
