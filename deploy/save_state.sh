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
# Another job (e.g. Retrain) may have pushed meanwhile: replay our commit on top and retry.
for attempt in 1 2 3; do
  git push -q origin HEAD:bot-state && exit 0
  git pull -q --rebase origin bot-state || true
done
echo "Could not save state after 3 tries" >&2
exit 1
