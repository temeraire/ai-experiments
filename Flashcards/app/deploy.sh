#!/bin/bash
# Deploy the flashcards app to the droplet, wired exactly like the other tools.
#
#   bash Flashcards/app/deploy.sh <ssh-target>
#   e.g.  bash Flashcards/app/deploy.sh root@165.227.29.79
#
# What it does (all remote edits are backed up first):
#   1. scp the app to /srv/flashcards
#   2. install + start a systemd service (127.0.0.1:8642)
#   3. clone the existing /teleprompter nginx location as /flashcards/
#      (inherits identical auth_request wiring); nginx -t gates the reload
#   4. clone the teleprompter card on portal.html into a Flashcards card
#   5. verify from inside and out
set -euo pipefail
TARGET=${1:?usage: bash deploy.sh user@host}
APP_DIR=$(cd "$(dirname "$0")" && pwd)

echo "== packaging =="
tar -czf /tmp/flashcards_app.tgz -C "$APP_DIR" server.js README.md public
scp /tmp/flashcards_app.tgz "$TARGET:/tmp/flashcards_app.tgz"

echo "== remote install =="
ssh "$TARGET" 'sudo bash -s' <<'REMOTE'
set -euo pipefail
STAMP=$(date +%Y%m%d_%H%M%S)

echo "--- 1. app files -> /srv/flashcards"
mkdir -p /srv/flashcards/data
tar -xzf /tmp/flashcards_app.tgz -C /srv/flashcards

echo "--- 2. systemd service"
# Reuse the node binary the auth gateway (or any existing service) runs with.
NODE_BIN=$(grep -h "ExecStart=.*node" /etc/systemd/system/*.service 2>/dev/null \
           | head -1 | sed 's/ExecStart=//' | awk '{print $1}')
NODE_BIN=${NODE_BIN:-$(command -v node)}
echo "    node: $NODE_BIN"
cat > /etc/systemd/system/flashcards.service <<UNIT
[Unit]
Description=Flashcards app (AB study deck)
After=network.target

[Service]
ExecStart=$NODE_BIN /srv/flashcards/server.js
Environment=PORT=8642
Environment=HOST=127.0.0.1
Environment=FLASHCARDS_DATA=/srv/flashcards/data/reviews.jsonl
Restart=always
User=root

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now flashcards
sleep 1
curl -sf http://127.0.0.1:8642/api/stats >/dev/null \
  && echo "    service OK (api/stats reachable)" \
  || { echo "!! service failed"; systemctl status flashcards --no-pager | tail -5; exit 1; }

echo "--- 3. nginx location (cloned from /teleprompter)"
CONF=$(grep -rl "tools.gregkiddfornevada.com" /etc/nginx/sites-enabled/ 2>/dev/null | head -1)
[ -z "$CONF" ] && CONF=$(grep -rl "tools.gregkiddfornevada.com" /etc/nginx/conf.d/ 2>/dev/null | head -1)
if [ -z "$CONF" ]; then
  echo "!! could not find nginx conf for tools.gregkiddfornevada.com — dumping sites-enabled:"
  ls /etc/nginx/sites-enabled/ /etc/nginx/conf.d/ 2>/dev/null
  exit 1
fi
echo "    conf: $CONF"
cp "$CONF" "$CONF.bak_$STAMP"
python3 - "$CONF" <<'PY'
import re, sys
conf = sys.argv[1]
s = open(conf).read()
if "/flashcards" in s:
    print("    flashcards location already present — skipping nginx edit")
    raise SystemExit
m = re.search(r'(location\s+[^{}]*teleprompter[^{}]*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', s)
if not m:
    print("!! no teleprompter location found to clone. Server block follows:")
    print(s)
    raise SystemExit(1)
blk = m.group(1)
new = re.sub(r'teleprompter[^\s;]*', 'flashcards/', blk, count=1)
new = re.sub(r'proxy_pass\s+http://127\.0\.0\.1:\d+[^;]*;',
             'proxy_pass http://127.0.0.1:8642/;', new)
new = re.sub(r'^location\s+\S+', 'location /flashcards/', new)
# keep every other directive (auth_request etc.) verbatim
s = s.replace(blk, blk + "\n\n    " + new, 1)
open(conf, "w").write(s)
print("    inserted /flashcards/ location (cloned auth directives)")
print("    --- new block ---")
print("    " + new.replace("\n", "\n    "))
PY
nginx -t && systemctl reload nginx && echo "    nginx reloaded" \
  || { echo "!! nginx -t failed — restoring backup"; cp "$CONF.bak_$STAMP" "$CONF"; nginx -t; exit 1; }

echo "--- 4. portal card"
PORTAL=$(find /srv /var/www /opt /root /home -maxdepth 4 -name portal.html 2>/dev/null | head -1)
if [ -z "$PORTAL" ]; then
  echo "    portal.html not found on droplet — skipping (tell Claude)"
else
  echo "    portal: $PORTAL"
  cp "$PORTAL" "$PORTAL.bak_$STAMP"
  python3 - "$PORTAL" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
if "flashcards" in s.lower():
    print("    flashcards card already present — skipping")
    raise SystemExit
m = re.search(r'(<a class="tool-card" href="[^"]*teleprompter[^"]*">.*?</a>)', s, re.S)
if not m:
    print("    !! teleprompter card not found; skipping portal edit")
    raise SystemExit
card = m.group(1)
new = card
new = re.sub(r'href="[^"]*"', 'href="https://tools.gregkiddfornevada.com/flashcards/"', new, count=1)
new = re.sub(r'&#\d+;', '&#127183;', new, count=1)             # icon -> playing card
new = re.sub(r'<h2>[^<]*</h2>', '<h2>Flashcards</h2>', new)
new = re.sub(r'<p>[^<]*</p>', '<p>Study deck with answer tracking</p>', new)
s = s.replace(card, card + "\n    " + new, 1)
open(p, "w").write(s)
print("    added Flashcards card to portal")
PY
fi

echo "--- 5. verify"
curl -s -o /dev/null -w "    internal http://127.0.0.1:8642/ -> %{http_code}\n" http://127.0.0.1:8642/
curl -s -o /dev/null -w "    public   https://tools.gregkiddfornevada.com/flashcards/ -> %{http_code} (302=login redirect is correct)\n" https://tools.gregkiddfornevada.com/flashcards/
echo "== REMOTE DONE =="
REMOTE

echo "== all done =="
echo "Visit: https://tools.gregkiddfornevada.com/flashcards/ (log in as usual)"
