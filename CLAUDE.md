## Standard Workflow
1. First think through the problem, read the codebase for relevant files, and write a plan to todo.md.
2. The plan should have a list of todo items that you can check off as you complete them
3. Before you begin working, check in with me and I will verify the plan.
4. Then, begin working on the todo items, marking them as complete as you go.
5. Please every step of the way just give me a high level explanation of what changes you made
6. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
7. Test your changes before marking tasks as complete. Don't just implement - verify that the feature works as expected.
8. Finally, add a review section to the todo.md file with a summary of the changes you made and any other relevant information.

## Git 
Before fixing bugs or adding features, always create a new git branch with a descriptive name based on the issue or feature. Commit your changes only to that branch. Once you're done, push the branch and create a pull request for review.

 ## Critical Rules for Code Changes

  ### NEVER remove existing features
  - When fixing bugs, ONLY fix the specific issue reported
  - DO NOT remove, consolidate, or "simplify" existing functionality
  - DO NOT remove UI elements (like multiple progress bars, comparison views, etc.)
  - If you think something should be refactored, ASK FIRST

  ### Before making changes
  1. Identify the MINIMAL change needed to fix the issue
  2. Preserve ALL existing features and UI elements
  3. If the fix might affect other features, explain the impact and ask for confirmation

  ### When fixing bugs
  - Fix ONLY what's broken
  - Keep all existing features intact
  - Don't "improve" or "optimize" unrelated code
  - Don't make architectural changes without explicit permission

  ### Examples of unacceptable changes
  - Combining multiple progress bars into one
  - Removing side-by-side model comparisons
  - Simplifying multi-model features into single-model
  - Removing any existing UI components
  - Changing the user experience without being asked

  ### If unsure
  Always err on the side of making the SMALLEST possible change. Ask "Will this fix affect any
  existing features?" If yes, explain and get approval first.