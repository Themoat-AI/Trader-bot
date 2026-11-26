190861 rows, 11 symbols, 2024-12-03 to 2026-11-26
Target: next 24 bars return > 0.60% (round-trip cost). Base rate 0.394

Walk-forward: logistic []
 fold test_start   test_end  base_rate  accuracy  always_no_acc    auc  logloss  logloss_base
    0 2025-09-18 2025-11-30     0.3743    0.5533         0.6257 0.5126   0.6830        0.6674
    1 2025-11-30 2026-02-10     0.3391    0.6381         0.6609 0.5524   0.6544        0.6535
    2 2026-02-10 2026-04-23     0.4076    0.5921         0.5924 0.5898   0.6685        0.6760
    3 2026-04-23 2026-07-05     0.3629    0.6270         0.6371 0.4641   0.6649        0.6588
    4 2026-07-05 2026-09-15     0.3833    0.6166         0.6167 0.5097   0.6713        0.6663
    5 2026-09-15 2026-11-26     0.3628    0.6420         0.6372 0.5209   0.6587        0.6577
Mean AUC 0.5249  (0.5 = coin flip)   log-loss 0.6668 vs base 0.6633

Walk-forward: mlp [16, 8]
 fold test_start   test_end  base_rate  accuracy  always_no_acc    auc  logloss  logloss_base
    0 2025-09-18 2025-11-30     0.3743    0.5866         0.6257 0.5133   0.6943        0.6674
    1 2025-11-30 2026-02-10     0.3391    0.6388         0.6609 0.5375   0.6655        0.6535
    2 2026-02-10 2026-04-23     0.4076    0.5935         0.5924 0.5455   0.6725        0.6760
    3 2026-04-23 2026-07-05     0.3629    0.6227         0.6371 0.4694   0.6695        0.6588
    4 2026-07-05 2026-09-15     0.3833    0.6167         0.6167 0.5098   0.6741        0.6663
    5 2026-09-15 2026-11-26     0.3628    0.6348         0.6372 0.5257   0.6584        0.6577
Mean AUC 0.5169  (0.5 = coin flip)   log-loss 0.6724 vs base 0.6633

Training final model on all history...
Saved /home/runner/work/Trader-bot/Trader-bot/state/models/crypto.npz
