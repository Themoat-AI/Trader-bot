#!/usr/bin/env bash
# Monthly retrain: download fresh data, retrain, backtest, restart the bot.
# Schedule with cron, e.g. 1st of each month at 03:17 UTC:
#   17 3 1 * * /home/trader/trader-bot/deploy/retrain.sh >> /home/trader/trader-bot/data/retrain.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m trader.data
python -m trader.train
python -m trader.backtest
sudo systemctl restart trader-bot
