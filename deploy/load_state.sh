#!/usr/bin/env bash
# Check out the bot's saved state (journal, model, charts) from the `bot-state`
# branch into ./state. On the very first run the branch doesn't exist yet, so an
# empty one is created.
set -euo pipefail
git config user.name "trader-bot"
git config user.email "trader-bot@users.noreply.github.com"
if git ls-remote --exit-code --heads origin bot-state >/dev/null 2>&1; then
  git fetch --depth=1 origin bot-state:refs/heads/bot-state
  git worktree add state bot-state
else
  git worktree add --orphan -b bot-state state
fi
mkdir -p state/data state/models state/reports
