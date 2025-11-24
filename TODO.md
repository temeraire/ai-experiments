- We currently have several models available that I'm running locally. I would like to add the following models:
1) The Claude models which are already there, but which I don't think are yet working I haven't added any money to Anthropic to pay for the api calls. Can you verify that that is the current situation? And how much do you suggest adding as a beginning amount?
2) I'd also like to add gemini-3 but am not sure how to do that. 
3) I'd like to add the "kimi-k2-thinking:cloud" model to those available from ollama

- UI changes: I'd like to separate the external models for which I'll be charged each time I use the api, from the local and ollama ones for which I will not be charged. Perhaps we do this just by displaying those that are charged in green: Light green (standing for money) if they're available, and then dark green when they're selected for the current conversation. Can we also get a reasonably accurate estimate of how much each conversation is costing at each turn?

- New feature: I'd like the llm, after it has run, to summarize its results and to place that summary at the top of the document that it outputs. This is so that I can see more easily what the differences might be from model to model. 

- Folder and file naming: I'm seeing folders inside the conversations folder with names like this: conv_20251115_112821_137e4cdc. I can see the date, which is fine, but I don't know what the rest of the name is. Can you tell me how it is named? I would like each folder and conversation document to include in its name the date and whichever model(s) are used. I'd also like to be able to see at a glance in the folder name how many models are contained with in it, which should reflect how many models were used. For this, I think we need an abbreviation system. Can you please invent one? If each model abbreviation is separated by an underscore or dash, I'll be able to see at a glance how many models were used.

On the same level as that folder, I see one called conv_20251122_115117_c94ae53a -- but it has nothing in it. In what circumstance does a folder get created with nothing in it? Is it because I didn't hit 'End & Save'?
Then, within that first folder conv_20251115_112821_137e4cdc, I see a subfolder called turn_002_Can-you-please-list-100-objects-or-items. The first part of that makes sense as it is the beginning of what I requested the app to do in the conversation. But I can't tell which model's response I'm looking at. 
- While it is good to be able to see these turn documents in multiple formats, I think it's currently unnecessary. For now, let's comment out the .md, .jsonl, and .html versions. Just output the .docx format.  And please keep the model name and turn number in the name. So instead of turn.docx, we have turn3_gemini3.docx (for example). Folder should also be identifiable with model abbreviations.

---

## IMPLEMENTATION PLAN

### Project Overview
This is a Flask-based web application that allows users to interact with multiple LLMs (both local via Ollama and external APIs like Claude). The app supports single model queries and multi-model comparison mode, saving conversations with multiple output formats.

### Current Analysis

#### 1. Claude API Status
- **API Key Found**: Yes, configured in .env file
- **Models Listed**: 6 Claude models already defined in llm_client.py
- **Recommendation for Credits**: Start with $20-50 for initial testing
  - Claude Sonnet 4.5: ~$3/$15 per million tokens (input/output)
  - Claude Opus 4.1: ~$15/$75 per million tokens
  - Claude Haiku 4.5: ~$0.80/$4 per million tokens

#### 2. Current Folder Naming
- **Format**: `conv_YYYYMMDD_HHMMSS_[8-char-uuid]`
- **Example**: `conv_20251115_112821_137e4cdc`
- The last part is a UUID for uniqueness

### Implementation Tasks

#### Phase 1: External Model Integration [COMPLETED]

##### 1.1 Verify Claude Models [COMPLETED]
- [x] Test Claude API connectivity with existing key - WORKING
- [x] Verify token counting is working - Already implemented in models.py
- [x] Add error handling for quota/rate limits - Already implemented
- [x] Test all 6 Claude models - API responds correctly

##### 1.2 Add Gemini Integration [COMPLETED]
- [x] Install Google SDK: `pip install google-generativeai`
- [x] GOOGLE_API_KEY already in .env (discovered during implementation)
- [x] Create gemini functions in llm_client.py:
  - `is_gemini_model()` - Added
  - `call_gemini()` - Added
  - `get_gemini_models()` - Added (5 models: gemini-2.5-flash/pro, 2.0-flash, 1.5-pro/flash)
- [x] Update `call_llm()` routing - Done
- [x] Update routes.py to include Gemini models in list - Done
- [x] Update config.py with HAS_GEMINI check - Done

##### 1.3 Add kimi-k2-thinking:cloud [PARTIAL]
- [x] Run: `ollama pull kimi-k2-thinking:cloud` - Done
- [x] Verify in ollama list - Shows as available
- [ ] Test functionality - BLOCKED: Requires Kimi API authentication (401 Unauthorized)
  - Note: This is a cloud-hosted model requiring Moonshot AI API credentials

#### Phase 2: UI Cost Visibility

##### 2.1 Model Color Coding
- [ ] Add model metadata with pricing info
- [ ] Update frontend.py with color coding:
  - Light green (#c8e6c9): Paid models available
  - Dark green (#2e7d32): Paid models selected
  - Normal: Free/local models
- [ ] Add $ icon for paid models

##### 2.2 Cost Estimation
- [ ] Create pricing configuration:
  ```python
  PRICING = {
      "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
      "claude-opus-4-1": {"input": 0.015, "output": 0.075},
      "claude-haiku-4-5": {"input": 0.0008, "output": 0.004},
      # Add Gemini pricing when available
  }
  ```
- [ ] Display cost per turn
- [ ] Show cumulative conversation cost
- [ ] Add to exports

#### Phase 3: File System Improvements

##### 3.1 Model Abbreviation System
```python
MODEL_ABBREVS = {
    # Claude
    "claude-sonnet-4-5-20250929": "CS45",
    "claude-opus-4-1-20250805": "CO41",
    "claude-haiku-4-5-20251015": "CH45",
    "claude-3-5-sonnet-20241022": "C35S",
    "claude-3-5-haiku-20241022": "C35H",

    # Gemini (to be added)
    "gemini-3-pro": "G3P",
    "gemini-3-flash": "G3F",

    # Ollama examples
    "qwen2.5:32b-instruct": "QW32",
    "llama3.1:70b": "LL70",
    "kimi-k2-thinking:cloud": "KK2T",
}
```

##### 3.2 New Folder Naming
- **Format**: `conv_YYYYMMDD_HHMMSS_[models]_[N]models`
- **Examples**:
  - `conv_20251123_143022_CS45_1model`
  - `conv_20251123_143022_CS45-CO41-QW32_3models`

##### 3.3 Turn Folder Naming
- **Single**: `turn_001_CS45_[prompt-slug]`
- **Comparison**: `turn_001_comparison_CS45-CO41_[prompt-slug]`

##### 3.4 Output Format Simplification
- [ ] Comment out .md generation
- [ ] Comment out .html generation
- [ ] Keep .jsonl for internal use only
- [ ] Rename .docx files: `turn1_CS45.docx`

#### Phase 4: Auto-Summary Feature

##### 4.1 Add Summary Generation
- [ ] Add summary prompt template
- [ ] Generate summary after each response
- [ ] Place at top of .docx file
- [ ] Display in UI (collapsible)
- [ ] Store in conversation data

#### Phase 5: Bug Fixes

##### 5.1 Empty Folder Issue
- [ ] Add auto-save on page unload
- [ ] Periodic auto-save (every 30 seconds)
- [ ] Cleanup empty folders on startup
- [ ] Add recovery for incomplete conversations

### Testing Checklist
- [ ] Claude models working with pricing
- [ ] Gemini models integrated
- [ ] kimi-k2-thinking:cloud available
- [ ] Cost tracking accurate
- [ ] New naming conventions applied
- [ ] Only .docx files generated
- [ ] Summaries generating
- [ ] No empty folders created

### Notes
- Follow CLAUDE.md guidelines: minimal changes, preserve features
- Create new git branch before starting
- Test incrementally after each phase
