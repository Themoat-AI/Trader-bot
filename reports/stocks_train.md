36406 rows, 11 symbols, 2025-01-03 to 2026-11-25
Target: next 24 bars return > 0.10% (round-trip cost). Base rate 0.516

Walk-forward: logistic []
 fold test_start   test_end  base_rate  accuracy  always_no_acc    auc  logloss  logloss_base
    0 2025-10-09 2025-12-16     0.5309    0.5111         0.4691 0.4577   0.7008        0.6932
    1 2025-12-17 2026-02-26     0.4565    0.4871         0.5435 0.5851   0.7093        0.7075
    2 2026-02-26 2026-05-05     0.5693    0.4967         0.4307 0.3993   0.7024        0.6856
    3 2026-05-06 2026-07-15     0.4993    0.5246         0.5007 0.5474   0.6936        0.6958
    4 2026-07-15 2026-09-21     0.4754    0.4644         0.5246 0.4671   0.7085        0.6991
    5 2026-09-21 2026-11-25     0.4029    0.4696         0.5971 0.5567   0.7061        0.7050
Mean AUC 0.5022  (0.5 = coin flip)   log-loss 0.7035 vs base 0.6977

Walk-forward: mlp [16, 8]
 fold test_start   test_end  base_rate  accuracy  always_no_acc    auc  logloss  logloss_base
    0 2025-10-09 2025-12-16     0.5309    0.5164         0.4691 0.4915   0.6994        0.6932
    1 2025-12-17 2026-02-26     0.4565    0.5099         0.5435 0.5904   0.6895        0.7075
    2 2026-02-26 2026-05-05     0.5693    0.5485         0.4307 0.5251   0.6892        0.6856
    3 2026-05-06 2026-07-15     0.4993    0.5213         0.5007 0.5606   0.6904        0.6958
    4 2026-07-15 2026-09-21     0.4754    0.4718         0.5246 0.4714   0.7182        0.6991
    5 2026-09-21 2026-11-25     0.4029    0.4687         0.5971 0.5111   0.7273        0.7050
Mean AUC 0.5250  (0.5 = coin flip)   log-loss 0.7023 vs base 0.6977

Training final model on all history...
Saved /home/runner/work/Trader-bot/Trader-bot/state/models/stocks.npz
