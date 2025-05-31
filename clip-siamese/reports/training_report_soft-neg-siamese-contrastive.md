# Обучение SiameseRuCLIP с простыми негативами

# General config

```
# NAME_MODEL_NAME = 'cointegrated/rubert-tiny' # 'DeepPavlov/distilrubert-tiny-cased-conversational-v1'
# DESCRIPTION_MODEL_NAME = 'cointegrated/rubert-tiny'
# PRELOAD_MODEL_NAME = None

NAME_MODEL_NAME = None
DESCRIPTION_MODEL_NAME = None
PRELOAD_MODEL_NAME = 'cc12m_rubert_tiny_ep_1.pt' # preload ruclip

DROPOUT = 0.5
# DROPOUT = None

BEST_CKPT_METRIC = 'f1'
# BEST_CKPT_METRIC = 'pos_acc'

MOMENTUM=0.9
WEIGHT_DECAY=1e-2
CONTRASTIVE_THRESHOLD=0.3

TEST_RATIO = 0.1
VAL_RATIO = 0.1

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED=42
```

# Results + specific configs

## 1. 50 positives ~ 30 sec per epoch

### 1.1 Default config [val F1-score @ 5 ep = 0.721]

```python
# GPU config
BATCH_SIZE_PER_DEVICE=60
EPOCHS=20
POS_NEG_RATIO=1.0
HARD_SOFT_RATIO=0.7
LIMIT_TRAIN_POS_PAIRS_PER_QUERY=50 # 50 for small GPU test
LIMIT_VAL_POS_PAIRS_PER_QUERY=None
LIMIT_TEST_POS_PAIRS_PER_QUERY=None
LIMIT_QUERIES = None
SAVE_EVERY_N_EPOCHS= 2
CONTRASTIVE_THRESHOLD=0.3
```

[val] Epoch 5 – loss: 0.6450, P Acc: 0.960, N Acc: 0.315, Avg Acc: 0.637, F1: 0.727, thr*: 0.430 (optimised: f1)

[val] Epoch 20 – loss: 0.5069, P Acc: 0.838, N Acc: 0.510, Avg Acc: 0.674, F1: 0.721, thr*: 0.678 (optimised: f1)

### 1.2 Pos/ Neg ratio 2x [val F1-score @ 5 ep = 0.574]

```python
# GPU config
BATCH_SIZE_PER_DEVICE=60
# EPOCHS=5 # 5 for rough estimate on validation
EPOCHS = 5 # 20 for convergence
POS_NEG_RATIO=2.0
HARD_SOFT_RATIO=0.7
LIMIT_TRAIN_POS_PAIRS_PER_QUERY=50 # 50 for small GPU test
LIMIT_VAL_POS_PAIRS_PER_QUERY=None
LIMIT_TEST_POS_PAIRS_PER_QUERY=None
LIMIT_QUERIES = None
SAVE_EVERY_N_EPOCHS= 2
```

[val] Epoch 5 – loss: 0.5811, P Acc: 0.864, N Acc: 0.415, Avg Acc: 0.639, F1: 0.574, thr*: 0.761 (optimised: f1)

### 1.3 Pos/Neg ratio 4x [val F1-score @ 5 ep = 0.485]

```python
# GPU config
BATCH_SIZE_PER_DEVICE=60
EPOCHS=5 # 5 for rough estimate on validation
# EPOCHS = 20 # 20 for convergence
POS_NEG_RATIO=4.0
HARD_SOFT_RATIO=0.7
LIMIT_TRAIN_POS_PAIRS_PER_QUERY=50 # 50 for small GPU test
LIMIT_VAL_POS_PAIRS_PER_QUERY=None
LIMIT_TEST_POS_PAIRS_PER_QUERY=None
LIMIT_QUERIES = None
SAVE_EVERY_N_EPOCHS= 2
```

[val] Epoch 5 – loss: 0.4742, P Acc: 0.737, N Acc: 0.670, Avg Acc: 0.704, F1: 0.485, thr*: 0.663 (optimised: f1)

### 1.4 Pos/Neg ratio 8x [val F1-score @ 5 ep = 0.35]

```python
# GPU config
BATCH_SIZE_PER_DEVICE=60
EPOCHS=5 # 5 for rough estimate on validation
# EPOCHS = 20 # 20 for convergence
POS_NEG_RATIO=8.0
HARD_SOFT_RATIO=0.7
LIMIT_TRAIN_POS_PAIRS_PER_QUERY=50 # 50 for small GPU test
LIMIT_VAL_POS_PAIRS_PER_QUERY=None
LIMIT_TEST_POS_PAIRS_PER_QUERY=None
LIMIT_QUERIES = None
SAVE_EVERY_N_EPOCHS= 2
```

[val] Epoch 5 – loss: 0.4407, P Acc: 0.918, N Acc: 0.644, Avg Acc: 0.781, F1: 0.702, thr*: 0.595 (optimised: f1)

### 1.3 Hard/Soft ratio 0.3 [val F1-score @ 5 ep = 0.702]

```python
# GPU config
BATCH_SIZE_PER_DEVICE=60
EPOCHS=5 # 5 for rough estimate on validation
# EPOCHS = 20 # 20 for convergence
POS_NEG_RATIO=2.0
HARD_SOFT_RATIO=0.3
LIMIT_TRAIN_POS_PAIRS_PER_QUERY=50 # 50 for small GPU test
LIMIT_VAL_POS_PAIRS_PER_QUERY=None
LIMIT_TEST_POS_PAIRS_PER_QUERY=None
LIMIT_QUERIES = None
SAVE_EVERY_N_EPOCHS= 2
```

[val] Epoch 5 – loss: 0.4407, P Acc: 0.918, N Acc: 0.644, Avg Acc: 0.781, F1: 0.702, thr*: 0.595 (optimised: f1)

### 1.3 Hard/Soft ratio 0.85 [val F1-score @ 5 ep = 0.787 🏆]

```python
# GPU config
BATCH_SIZE_PER_DEVICE=60
EPOCHS=5 # 5 for rough estimate on validation
# EPOCHS = 20 # 20 for convergence
POS_NEG_RATIO=2.0
HARD_SOFT_RATIO=0.8
LIMIT_TRAIN_POS_PAIRS_PER_QUERY=50 # 50 for small GPU test
LIMIT_VAL_POS_PAIRS_PER_QUERY=None
LIMIT_TEST_POS_PAIRS_PER_QUERY=None
LIMIT_QUERIES = None
SAVE_EVERY_N_EPOCHS= 2
```

[val] Epoch 5 – loss: 0.5040, P Acc: 0.842, N Acc: 0.677, Avg Acc: 0.760, F1: 0.787, thr*: 0.369 (optimised: f1)

## 2. 500 positives ~ 6 min per epoch

### 2.1 Default config [val F1-score @ 4 ep = 0.817]

```python
# GPU config
BATCH_SIZE_PER_DEVICE = 60
EPOCHS = 20
POS_NEG_RATIO = 1.0
HARD_SOFT_RATIO = 0.7
LIMIT_TRAIN_POS_PAIRS_PER_QUERY = 500  # 50 for small GPU test
LIMIT_VAL_POS_PAIRS_PER_QUERY = None
LIMIT_TEST_POS_PAIRS_PER_QUERY = None
LIMIT_QUERIES = None
SAVE_EVERY_N_EPOCHS = 2
```

[val] Epoch 1 – loss: 0.4807, P Acc: 0.840, N Acc: 0.536, Avg Acc: 0.688, F1: 0.730, thr*: 0.739 (optimised: f1)

[val] Epoch 2 – loss: 0.3731, P Acc: 0.941, N Acc: 0.573, Avg Acc: 0.757, F1: 0.795, thr*: 0.950 (optimised: f1)

[val] Epoch 3 – loss: 0.3543, P Acc: 0.900, N Acc: 0.671, Avg Acc: 0.786, F1: 0.808, thr*: 0.995 (optimised: f1)

[val] Epoch 4 – loss: 0.3852, P Acc: 0.912, N Acc: 0.676, Avg Acc: 0.794, F1: 0.817, thr*: 1.319 (optimised: f1)

## 3. 5000 positives ~ 90 min per epoch

### 2.1 Default config [val F1-score @ 1 ep = 0.829]

```python
# GPU config
BATCH_SIZE_PER_DEVICE = 60
EPOCHS = 20
POS_NEG_RATIO = 2.0
HARD_SOFT_RATIO = 0.6
LIMIT_TRAIN_POS_PAIRS_PER_QUERY = 5000
LIMIT_VAL_POS_PAIRS_PER_QUERY = None
LIMIT_TEST_POS_PAIRS_PER_QUERY = None
LIMIT_QUERIES = None
SAVE_EVERY_N_EPOCHS = 2
```

[val] Epoch 1 – loss: 0.4558, P Acc: 0.802, N Acc: 0.932, Avg Acc: 0.867, F1: 0.829, thr*: 1.010 (optimised: f1)

[val] Epoch 2 – loss: 1.5096, P Acc: 0.713, N Acc: 0.951, Avg Acc: 0.832, F1: 0.788, thr*: 0.769 (optimised: f1)

[val] Epoch 3 – loss: 1.8986, P Acc: 0.752, N Acc: 0.937, Avg Acc: 0.844, F1: 0.802, thr*: 1.101 (optimised: f1)

[val] Epoch 4 – loss: 1.8526, P Acc: 0.650, N Acc: 0.964, Avg Acc: 0.807, F1: 0.756, thr*: 0.362 (optimised: f1)

[val] Epoch 5 – loss: 0.8128, P Acc: 0.727, N Acc: 0.936, Avg Acc: 0.832, F1: 0.785, thr*: 1.085 (optimised: f1)

### 2.2 Low LR, low patience, adapted margin [val F1-score @ 6 ep = 0.980 🏆]

```python
# GPU config (large)
# TODO: introduce gradient accumulation
BATCH_SIZE_PER_DEVICE=60
EPOCHS=10
POS_NEG_RATIO=1.0
HARD_SOFT_RATIO=0.80
LIMIT_TRAIN_POS_PAIRS_PER_QUERY=5000
LIMIT_VAL_POS_PAIRS_PER_QUERY=None
LIMIT_TEST_POS_PAIRS_PER_QUERY=None
LIMIT_QUERIES = None
SHEDULER_PATIENCE=1 # in epochs
LR=3e-5
CONTRASTIVE_MARGIN=1.0
```

[val] Epoch 1 – loss: 0.0607, P Acc: 0.983, N Acc: 0.872, Avg Acc: 0.928, F1: 0.934, thr*: 0.266 (optimised: f1)
[val] Epoch 2 – loss: 0.0508, P Acc: 0.985, N Acc: 0.919, Avg Acc: 0.952, F1: 0.955, thr*: 0.226 (optimised: f1)

[val] Epoch 3 – loss: 0.0389, P Acc: 0.980, N Acc: 0.964, Avg Acc: 0.972, F1: 0.973, thr*: 0.236 (optimised: f1)

[val] Epoch 4 – loss: 0.0287, P Acc: 0.983, N Acc: 0.973, Avg Acc: 0.978, F1: 0.979, thr*: 0.312 (optimised: f1)

[val] Epoch 5 – loss: 0.0293, P Acc: 0.981, N Acc: 0.974, Avg Acc: 0.978, F1: 0.978, thr*: 0.226 (optimised: f1)
[val] Epoch 6 – loss: 0.0249, P Acc: 0.986, N Acc: 0.972, Avg Acc: 0.979, F1: 0.980, thr*: 0.397 (optimised: f1)

[val] Epoch 7 – loss: 0.0279, P Acc: 0.981, N Acc: 0.975, Avg Acc: 0.978, F1: 0.978, thr*: 0.307 (optimised: f1)

[val] Epoch 8 – loss: 0.0259, P Acc: 0.984, N Acc: 0.973, Avg Acc: 0.979, F1: 0.979, thr*: 0.372 (optimised: f1)