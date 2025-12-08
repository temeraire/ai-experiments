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
