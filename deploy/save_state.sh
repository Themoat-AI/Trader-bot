#!/usr/bin/env bash
# Commit the bot's state back to the `bot-state` branch so the next run picks it up.
# Only small files are saved; price data is re-downloaded each time.
set -euo pipefail
message="${1:-Update bot state}"
cd state
for path in data/journal.db data/bot.log models reports/*.png reports/*.md; do
  [ -e "$path" ] && git add -f "$path"
done
if git diff --cached --quiet; then
  echo "No state changes to save."
  exit 0
fi
git commit -q -m "$message"
git push origin HEAD:bot-state
