#!/bin/bash
# Open an iTerm2 window with one pane per project agent, each live-tailing that
# agent's transcript as it works. Run from anywhere:
#     bash alien_baby/utils/agents_dashboard.sh
#
# Layout (2 columns x 3 rows, 6 panes):
#   experiment-strategist | training-engineer
#   results-analyst       | theory-monitor
#   literature-scout      | ALL agents (catch-all stream)
#
# Requires iTerm2. macOS may prompt once to allow iTerm2 automation.

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WATCH="python3 $REPO/alien_baby/utils/watch_agents.py"

osascript <<EOF
tell application "iTerm2"
  activate
  set newWindow to (create window with default profile)
  tell newWindow
    -- column 1, rows via horizontal splits; column 2 by vertical splits per row
    tell current session of current tab
      set name to "strategist"
      write text "$WATCH --role experiment-strategist"
      set c1r2 to (split horizontally with default profile)
      set c2r1 to (split vertically with default profile)
    end tell
    tell c2r1
      set name to "trainer"
      write text "$WATCH --role training-engineer"
    end tell
    tell c1r2
      set name to "analyst"
      write text "$WATCH --role results-analyst"
      set c1r3 to (split horizontally with default profile)
      set c2r2 to (split vertically with default profile)
    end tell
    tell c2r2
      set name to "theory"
      write text "$WATCH --role theory-monitor"
    end tell
    tell c1r3
      set name to "lit-scout"
      write text "$WATCH --role literature-scout"
      set c2r3 to (split vertically with default profile)
    end tell
    tell c2r3
      set name to "ALL"
      write text "$WATCH --all"
    end tell
  end tell
end tell
EOF
