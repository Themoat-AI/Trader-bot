#!/usr/bin/env bash
# Trade every hourly bar for up to TRADE_LOOP_MINUTES (default 5.5 hours, under
# GitHub's 6-hour job limit). GitHub's scheduler often skips runs, so one run
# keeps the bot going for hours and the next queued run picks up after it.
#
# Usage: deploy/trade_loop.sh [--live]
set -uo pipefail
live_flag="${1:-}"
end=$(( $(date +%s) + ${TRADE_LOOP_MINUTES:-330} * 60 ))

while :; do
  # Pick up models saved by the Retrain workflow since this run started.
  git -C state pull -q --rebase origin bot-state 2>/dev/null || true

  for u in $TRADER_UNIVERSES; do
    if [ ! -f "state/models/$u.npz" ]; then
      echo "::group::Training first model for $u"
      python -m trader.data --universe "$u" && python -m trader.train --universe "$u" \
        && python -m trader.backtest --universe "$u" || echo "::warning::Training $u failed"
      echo "::endgroup::"
    fi
    if [ -f "state/models/$u.npz" ]; then
      python -m trader.bot --once --universe "$u" $live_flag || echo "::warning::$u cycle failed"
    fi
  done
  deploy/save_state.sh "Bot cycle $(date -u +%Y-%m-%dT%H:%MZ)" || echo "::warning::Could not save state"

  now=$(date +%s)
  next=$(( (now / 3600 + 1) * 3600 + 120 ))   # 2 minutes after the next hourly bar closes
  if [ "$next" -ge "$end" ]; then
    echo "Loop time used up; the next scheduled run continues."
    break
  fi
  echo "Next cycle at $(date -u -d @$next +%H:%M) UTC"
  sleep $(( next - now ))
done
