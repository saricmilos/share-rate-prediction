# Results

Written by `notebooks/05_results.ipynb` from its own runs; every number below is in `results.json` beside this file. Intervals are 95%, from 2,000 draws of creators (seed 2026), paired where they are gains. Lift is log10 of the share rate minus the creator's recent median, so an MAE of 0.37 is a factor of 10^0.37 = 2.3 either way. A gain is the base's error minus the model's: positive is better.

## The result, on the test split (opened once)

The final model is **metadata, the words and recent form (the combined model)**. On the newest videos (posted 1 April 2026 to 18 September 2026), for creators it trained on (12,455 videos from 138 creators), predicting zero lift misses by 0.3752 log10, a factor of 2.37 either way. The final model misses by 0.3595: a gain of +0.0157 [+0.0109, +0.0210], an error 4.2% smaller. For creators it never saw (9,285 videos from 81 creators), zero lift misses by 0.3774 and the final model gains +0.0123 [+0.0069, +0.0181] (3.3%).

### What each part adds, on validation and on the test

Each row is one model's gain over the model before it, in MAE, seen creators unless the column says otherwise.

| Claim | Model over base | Validation, seen | Test, seen | Test, seen, no reposts | Test, unseen | Holds on test (seen / unseen) |
|---|---|---|---|---|---|---|
| The final model beats zero lift | combined over B0 zero lift | +0.0159 [+0.0108, +0.0206] | +0.0157 [+0.0109, +0.0210] | +0.0157 [+0.0108, +0.0210] | +0.0123 [+0.0069, +0.0181] | yes / yes |
| Metadata beats zero lift | B3 ridge over B0 zero lift | +0.0007 [+0.0002, +0.0011] | +0.0005 [+0.0000, +0.0011] | +0.0005 [+0.0000, +0.0011] | +0.0003 [-0.0002, +0.0009] | yes / no |
| The words add to the metadata | 4B ridge over B3 ridge | +0.0097 [+0.0062, +0.0133] | +0.0098 [+0.0062, +0.0133] | +0.0098 [+0.0062, +0.0133] | +0.0064 [+0.0029, +0.0100] | yes / yes |
| Recent form adds to the metadata and the words | combined over 4B ridge | +0.0055 [+0.0032, +0.0080] | +0.0054 [+0.0027, +0.0086] | +0.0053 [+0.0027, +0.0086] | +0.0056 [+0.0025, +0.0088] | yes / yes |
| The frame adds to the metadata and the words | image ridge over 4B ridge (rows with a frame) | +0.0003 [-0.0002, +0.0008] | +0.0004 [-0.0002, +0.0009] | +0.0004 [-0.0002, +0.0009] | +0.0004 [-0.0006, +0.0014] | no / no |

### Every model on the test

**Creators the model trained on** (12,455 videos, 138 creators). Each cell: the value, then its gain over zero lift [interval].

| Model | MAE | skill | Spearman inside a creator | ROC-AUC | PR-AUC | top-10% share |
|---|---|---|---|---|---|---|
| zero lift (the creator's recent median) | 0.3752 +0.0000 [+0.0000, +0.0000] | 0.0000 +0.0000 [+0.0000, +0.0000] | 0.000 +0.000 [+0.000, +0.000] | 0.500 +0.000 [+0.000, +0.000] | 0.247 +0.000 [+0.000, +0.000] | 0.253 +0.000 [+0.000, +0.000] |
| metadata (B3 ridge) | 0.3747 +0.0005 [+0.0000, +0.0011] | 0.0047 +0.0047 [+0.0025, +0.0070] | 0.078 +0.078 [+0.046, +0.115] | 0.537 +0.037 [+0.016, +0.060] | 0.266 +0.019 [+0.007, +0.035] | 0.295 +0.042 [+0.004, +0.081] |
| recent form (B1) | 0.3719 +0.0033 [+0.0004, +0.0067] | 0.0357 +0.0357 [+0.0222, +0.0506] | 0.153 +0.153 [+0.117, +0.193] | 0.597 +0.097 [+0.076, +0.125] | 0.339 +0.092 [+0.071, +0.118] | 0.368 +0.115 [+0.080, +0.152] |
| metadata and the words (4B ridge) | 0.3649 +0.0103 [+0.0068, +0.0138] | 0.0556 +0.0556 [+0.0420, +0.0703] | 0.244 +0.244 [+0.208, +0.279] | 0.591 +0.091 [+0.067, +0.115] | 0.319 +0.072 [+0.047, +0.102] | 0.394 +0.141 [+0.100, +0.185] |
| metadata, the words and the first frame (image ridge) | 0.3645 +0.0107 [+0.0068, +0.0145] | 0.0575 +0.0575 [+0.0426, +0.0728] | 0.253 +0.253 [+0.219, +0.287] | 0.590 +0.090 [+0.065, +0.114] | 0.318 +0.071 [+0.047, +0.100] | 0.394 +0.141 [+0.101, +0.183] |
| metadata, the words and recent form (the combined model) | 0.3595 +0.0157 [+0.0109, +0.0210] | 0.0904 +0.0904 [+0.0705, +0.1113] | 0.303 +0.303 [+0.270, +0.338] | 0.633 +0.133 [+0.114, +0.154] | 0.365 +0.118 [+0.095, +0.148] | 0.435 +0.182 [+0.140, +0.227] |
| the words' breakout head (4B logistic, ranking only) | — | — | 0.228 +0.228 [+0.194, +0.263] | 0.589 +0.089 [+0.071, +0.110] | 0.321 +0.074 [+0.053, +0.099] | 0.380 +0.127 [+0.091, +0.167] |

**Creators it never saw** (9,285 videos, 81 creators). Each cell: the value, then its gain over zero lift [interval].

| Model | MAE | skill | Spearman inside a creator | ROC-AUC | PR-AUC | top-10% share |
|---|---|---|---|---|---|---|
| zero lift (the creator's recent median) | 0.3774 +0.0000 [+0.0000, +0.0000] | 0.0000 +0.0000 [+0.0000, +0.0000] | 0.000 +0.000 [+0.000, +0.000] | 0.500 +0.000 [+0.000, +0.000] | 0.247 +0.000 [+0.000, +0.000] | 0.236 +0.000 [+0.000, +0.000] |
| metadata (B3 ridge) | 0.3771 +0.0003 [-0.0002, +0.0009] | 0.0027 +0.0027 [-0.0008, +0.0059] | 0.042 +0.042 [-0.002, +0.089] | 0.544 +0.044 [+0.020, +0.062] | 0.272 +0.025 [+0.008, +0.042] | 0.262 +0.026 [-0.013, +0.067] |
| recent form (B1) | 0.3731 +0.0043 [+0.0008, +0.0079] | 0.0356 +0.0356 [+0.0211, +0.0475] | 0.118 +0.118 [+0.074, +0.159] | 0.590 +0.090 [+0.068, +0.115] | 0.337 +0.090 [+0.055, +0.129] | 0.323 +0.087 [+0.044, +0.128] |
| metadata and the words (4B ridge) | 0.3707 +0.0067 [+0.0030, +0.0104] | 0.0402 +0.0402 [+0.0219, +0.0550] | 0.172 +0.172 [+0.124, +0.218] | 0.592 +0.092 [+0.069, +0.111] | 0.315 +0.068 [+0.048, +0.089] | 0.334 +0.098 [+0.055, +0.142] |
| metadata, the words and the first frame (image ridge) | 0.3703 +0.0071 [+0.0027, +0.0116] | 0.0429 +0.0429 [+0.0210, +0.0611] | 0.182 +0.182 [+0.135, +0.229] | 0.594 +0.094 [+0.071, +0.113] | 0.316 +0.069 [+0.048, +0.091] | 0.335 +0.100 [+0.059, +0.142] |
| metadata, the words and recent form (the combined model) | 0.3650 +0.0123 [+0.0069, +0.0181] | 0.0781 +0.0781 [+0.0531, +0.1008] | 0.220 +0.220 [+0.177, +0.261] | 0.636 +0.136 [+0.114, +0.155] | 0.374 +0.127 [+0.094, +0.163] | 0.383 +0.148 [+0.103, +0.191] |
| the words' breakout head (4B logistic, ranking only) | — | — | 0.164 +0.164 [+0.119, +0.206] | 0.585 +0.085 [+0.060, +0.107] | 0.311 +0.064 [+0.044, +0.086] | 0.316 +0.080 [+0.038, +0.125] |

## Validation: every model, three ways

The numbers every choice was made on, against zero lift, MAE and Spearman inside a creator.

| Model | Seen: MAE gain | Seen: Spearman | New: MAE gain | Held out: MAE gain |
|---|---|---|---|---|
| zero lift (the creator's recent median) | +0.0000 [+0.0000, +0.0000] | 0.000 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| metadata (B3 ridge) | +0.0007 [+0.0002, +0.0011] | 0.071 | -0.0004 [-0.0030, +0.0009] | +0.0009 [-0.0000, +0.0017] |
| recent form (B1) | +0.0025 [-0.0005, +0.0053] | 0.146 | +0.0019 [-0.0058, +0.0073] | +0.0005 [-0.0101, +0.0077] |
| metadata and the words (4B ridge) | +0.0104 [+0.0069, +0.0141] | 0.224 | +0.0100 [+0.0029, +0.0172] | +0.0098 [+0.0046, +0.0153] |
| metadata, the words and the first frame (image ridge) | +0.0107 [+0.0070, +0.0145] | 0.229 | +0.0099 [+0.0020, +0.0176] | +0.0106 [+0.0046, +0.0168] |
| metadata, the words and recent form (the combined model) | +0.0159 [+0.0108, +0.0206] | 0.275 | +0.0133 [+0.0014, +0.0228] | +0.0148 [+0.0035, +0.0237] |
| whether the platform kept the clip (a leak, never a model) | +0.0191 [+0.0114, +0.0282] | 0.521 | +0.0284 [+0.0000, +0.0583] | +0.0131 [+0.0015, +0.0282] |

## The ablation (validation, seen creators, the stacked combiner refitted on each subset)

| Parts | MAE | Against the combined model |
|---|---|---|
| nothing (zero lift) | 0.3687 | -0.0159 [-0.0206, -0.0108] |
| B3 | 0.3682 | -0.0154 [-0.0201, -0.0103] |
| the words | 0.3593 | -0.0065 [-0.0088, -0.0045] |
| recent form | 0.3650 | -0.0122 [-0.0166, -0.0077] |
| B3 + the words | 0.3589 | -0.0061 [-0.0084, -0.0039] |
| B3 + recent form | 0.3642 | -0.0114 [-0.0157, -0.0070] |
| the words + recent form | 0.3535 | -0.0007 [-0.0013, -0.0002] |
| B3 + the words + recent form | 0.3528 | +0.0000 [+0.0000, +0.0000] |

## Where it works (test, seen creators)

The final model's MAE gain over zero lift in each group, as a share of zero lift's error there. Groups of fewer than 5 creators are not shown.

**Creator size (median views before the test)**

| Group | Videos | Creators | Zero lift's MAE | Gain | Share of the error |
|---|---|---|---|---|---|
| under 23k | 5,051 | 35 | 0.3775 | +0.0109 [+0.0004, +0.0189] | 2.9% |
| 23k to 99k | 2,104 | 34 | 0.3675 | +0.0180 [+0.0081, +0.0303] | 4.9% |
| 99k to 508k | 3,885 | 34 | 0.3904 | +0.0203 [+0.0129, +0.0318] | 5.2% |
| over 508k | 1,415 | 35 | 0.3368 | +0.0166 [+0.0083, +0.0245] | 4.9% |

**Video length**

| Group | Videos | Creators | Zero lift's MAE | Gain | Share of the error |
|---|---|---|---|---|---|
| under 10 s | 4,590 | 100 | 0.3805 | +0.0168 [+0.0115, +0.0226] | 4.4% |
| 10 to 20 s | 4,134 | 121 | 0.3630 | +0.0145 [+0.0084, +0.0214] | 4.0% |
| 20 to 40 s | 2,570 | 123 | 0.3730 | +0.0151 [+0.0035, +0.0272] | 4.1% |
| 40 to 60 s | 1,116 | 103 | 0.4050 | +0.0162 [+0.0060, +0.0281] | 4.0% |
| over 60 s | 45 | 24 | 0.3444 | +0.0218 [-0.0157, +0.0600] | 6.3% |

**Talking or not**

| Group | Videos | Creators | Zero lift's MAE | Gain | Share of the error |
|---|---|---|---|---|---|
| talking | 9,655 | 138 | 0.3779 | +0.0167 [+0.0110, +0.0229] | 4.4% |
| text on screen, no speech | 2,270 | 94 | 0.3711 | +0.0139 [+0.0080, +0.0193] | 3.7% |
| neither | 530 | 77 | 0.3431 | +0.0044 [-0.0024, +0.0151] | 1.3% |

**Earlier videos in the library**

| Group | Videos | Creators | Zero lift's MAE | Gain | Share of the error |
|---|---|---|---|---|---|
| under 100 | 169 | 14 | 0.4535 | +0.0343 [-0.0212, +0.0734] | 7.6% |
| 100 to 300 | 652 | 29 | 0.3672 | +0.0069 [-0.0044, +0.0204] | 1.9% |
| 300 to 1,000 | 5,057 | 64 | 0.3619 | +0.0112 [+0.0049, +0.0181] | 3.1% |
| 1,000 or more | 6,577 | 43 | 0.3842 | +0.0195 [+0.0135, +0.0273] | 5.1% |

## Stage 1, for the page

From `notebooks/01_eda_target.ipynb`'s printed outputs: the creator explains 38% of the variance in log share rate; the same creators' median share rate rose 4.7× between their 2022 and 2026 videos; kept clips beat their creator's median 64% of the time, retired ones 43%. 143,489 videos from 231 creators have a target, and 24.5% broke out.
