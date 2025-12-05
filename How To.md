How To.md

Kill off and then restart app:
⏺ Bash(lsof -ti :5005 | xargs kill -9 2>/dev/null; sleep 1; python main.py)