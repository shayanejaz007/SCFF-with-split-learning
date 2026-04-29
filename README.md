# SCFF Split Learning — CIFAR-10
## Self-Supervised Contrastive Forward-Forward with Split & Multi-Client Training

---

## Table of Contents

1. [What is SCFF?](#1-what-is-scff)
2. [Architecture Overview](#2-architecture-overview)
3. [Where the Split Happens](#3-where-the-split-happens)
4. [How Client & Server Are Initialized](#4-how-client--server-are-initialized)
5. [How Activations Are Passed](#5-how-activations-are-passed)
6. [How Gradients Are Handled](#6-how-gradients-are-handled)
7. [Multi-Client Setup](#7-multi-client-setup)
   - [Does Raw Data Leave the Client?](#does-raw-data-leave-the-client)
   - [How Gradients Flow in Multi-Client](#how-gradients-flow-in-multi-client)
   - [The Weight Relay Mechanism](#the-weight-relay-mechanism)
8. [Evaluation & Linear Readout](#8-evaluation--linear-readout)
9. [How Accuracy Was Achieved](#9-how-accuracy-was-achieved)
10. [Running the Scripts](#10-running-the-scripts)
11. [Config Reference](#11-config-reference)

---

## 1. What is SCFF?

**SCFF** = **Self-supervised Contrastive Forward-Forward**.

It replaces backpropagation with a **layer-wise local loss** — the Forward-Forward algorithm (Hinton, 2022) adapted with a contrastive self-supervised objective.

### Core Idea

Instead of a global loss sent backward through the whole network, each layer trains independently using its own **goodness signal**:

- **Positive data** = real image pairs (same image, two augmented views)
- **Negative data** = mismatched pairs (one image channel-concatenated with a *different* image)

Each layer maximizes goodness on positives and minimizes goodness on negatives via the **softplus loss**:

```
L = log(1 + exp(a * (-goodness_pos + θ₁)))    ← push pos goodness above θ₁
  + log(1 + exp(b * (goodness_neg  - θ₂)))    ← push neg goodness below θ₂
  + λ * ||goodness_pos||₂                     ← regularise pos goodness magnitude
```

Where **goodness** = mean of squared ReLU activations over spatial dims.

Because each layer trains on `.detach()`-ed inputs, **no gradient ever crosses a layer boundary** — split learning is a natural fit.

---

## 2. Architecture Overview

The network has **3 convolutional layers** (NL=3), each a custom `Conv2d` block:

```
Input (3×32×32 CIFAR-10 image)
        │
  ┌─────▼─────┐
  │  Layer 0  │  ← CLIENT holds this
  │  Conv2d   │    5×5 kernel, 3→96 channels
  │  Triangle │    MaxPool(4, stride=2)
  └─────┬─────┘
        │  activations (detached tensor, [B, 96, H', W'])
        │  ← SPLIT BOUNDARY — only activations cross here, never raw data
  ┌─────▼─────┐
  │  Layer 1  │  ← SERVER holds this
  │  Conv2d   │    3×3 kernel, 96→384 channels
  │  Triangle │    MaxPool(4, stride=2)
  └─────┬─────┘
        │  activations (detached)
  ┌─────▼─────┐
  │  Layer 2  │  ← SERVER holds this
  │  Conv2d   │    3×3 kernel, 384→1536 channels
  │  ReLU     │    AvgPool(2, stride=2)
  └─────┬─────┘
        │
  Linear Readout (Dropout(0.2) → Linear → 10 classes)
```

**Layer summary:**

| Layer | Location | Ch In→Out | Kernel | Activation | Pool |
|-------|----------|-----------|--------|------------|------|
| 0     | Client   | 3 → 96    | 5×5    | Triangle   | MaxPool(4,2) |
| 1     | Server   | 96 → 384  | 3×3    | Triangle   | MaxPool(4,2) |
| 2     | Server   | 384→1536  | 3×3    | ReLU       | AvgPool(2,2) |

---

## 3. Where the Split Happens

**Split point: after Layer 0, before Layer 1.**

The hard cut is the `.detach()` call at the end of `SplitClient.train_step()`.

**`SCFF_CIFAR_Split.py` — Lines 363–364:**
```python
# After Layer 0 trains locally, activations are detached before going to server
x_pos_pooled = self.pool(self.net.act(out_pos)).detach()   # line 363
x_neg_pooled = self.pool(self.net.act(out_neg)).detach()   # line 364
```

From this point there is **zero computational graph** connecting client and server. The server receives plain tensors with no gradient history.

---

## 4. How Client & Server Are Initialized

### Positive/Negative Pair Construction

Before any layer runs, raw images are turned into pos/neg pairs. This is the contrastive mechanism.

**`SCFF_CIFAR_Split.py` — Lines 134–144 (`get_pos_neg_batch_imgcats`):**
```python
def get_pos_neg_batch_imgcats(batch_pos1, batch_pos2, p=1):
    batch_size = len(batch_pos1)
    batch_pos = torch.cat((batch_pos1, batch_pos2), dim=1)          # pos: same image, channel-cat
    random_indices = (torch.randperm(batch_size - 1) + 1)[:min(p, batch_size - 1)]
    labeles = torch.arange(batch_size)
    batch_negs = []
    for i in random_indices:
        batch_neg = batch_pos2[(labeles + i) % batch_size]           # neg: shifted (different) image
        batch_neg = torch.cat((batch_pos1, batch_neg), dim=1)
        batch_negs.append(batch_neg)
    return batch_pos, torch.cat(batch_negs)
```

### The Conv2d Block

Every layer uses the same custom `Conv2d` block with optional `concat` mode.

**`SCFF_CIFAR_Split.py` — Lines 185–230 (`class Conv2d`):**
```python
class Conv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, pad=0,
                 batchnorm=False, normdims=[1, 2, 3], norm="stdnorm", bias=True,
                 dropout=0.0, padding_mode="reflect", concat=True, act="relu"):
        ...
        self.conv_layer = nn.Conv2d(in_channels=input_channels,
                                    out_channels=output_channels,
                                    kernel_size=kernel_size, bias=bias)
        init.xavier_uniform_(self.conv_layer.weight)   # Xavier init

    def forward(self, x):
        x = self.bn1(x)
        x = F.pad(x, self.F_padding, self.padding_mode)
        x = self.norm(x)                               # stdnorm before conv
        if self.concat:
            lenchannel = x.size(1) // 2
            out = self.conv_layer(x[:, :lenchannel]) + self.conv_layer(x[:, lenchannel:])
        else:
            out = self.conv_layer(x)
        return out
```
> **`concat=True`** (Layer 0, Layer 2): splits the channel-concatenated pos/neg input in half, runs the same conv on each half, and sums them — this is the FF contrastive mechanism.
> **`concat=False`** (Layer 1): treats the full input as-is.

### Client Initialization

**`SCFF_CIFAR_Split.py` — Lines 313–320 (`class SplitClient.__init__`):**
```python
class SplitClient:
    """Client holds Layer 0 and trains it on raw images."""

    def __init__(self, net, pool, extra_pool, optimizer, scheduler, device):
        self.net = net             # Conv2d(3→96, 5×5, concat=True, act=triangle)
        self.pool = pool           # MaxPool2d(4, stride=2)
        self.extra_pool = extra_pool  # AvgPool2d(2, stride=2) — used at eval only
        self.optimizer = optimizer # AdamW(lr=0.02, weight_decay=0.0001)
        self.scheduler = scheduler # ExponentialLR(gamma=0.99)
        self.device = device
```

Built in main via `create_layer` using `layer_configs[0]` from `config.json`:

**`SCFF_CIFAR_Split.py` — Lines 283–310 (`create_layer`):**
```python
def create_layer(layer_config, opt_config, load_params, device, act):
    net = Conv2d(layer_config["ch_in"],       # 3
                 layer_config["channels"],     # 96
                 (layer_config["kernel_size"], layer_config["kernel_size"]),  # 5×5
                 pad=layer_config["pad"],      # 2
                 norm="stdnorm",
                 padding_mode=layer_config["padding_mode"],  # "reflect"
                 act=act)                      # "triangle"
    ...
    pool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, ceil_mode=True)
    extra_pool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
    optimizer = AdamW(net.parameters(), lr=opt_config["lr"],
                      weight_decay=opt_config["weight_decay"])
    scheduler = ExponentialLR(optimizer, opt_config["gamma"])
    return net, pool, extra_pool, optimizer, scheduler
```

### Server Initialization

**`SCFF_CIFAR_Split.py` — Lines 401–413 (`class SplitServer.__init__`):**
```python
class SplitServer:
    """Server holds Layers 1..NL-1 and trains them on activations from client."""

    def __init__(self, nets, pools, extra_pools, optimizers, schedulers, device):
        self.nets = nets             # [Conv2d(96→384), Conv2d(384→1536)]
        self.pools = pools           # [MaxPool(4,2), AvgPool(2,2)]
        self.extra_pools = extra_pools
        self.optimizers = optimizers # [AdamW(lr=0.001), AdamW(lr=0.0004)]
        self.schedulers = schedulers
        self.device = device
```

Built in main with a loop over `layer_configs[1:]` — one `create_layer` call per server layer.

---

## 5. How Activations Are Passed

### Client `train_step` — the full flow

**`SCFF_CIFAR_Split.py` — Lines 324–369:**
```python
def train_step(self, x, threshold1, threshold2, a, b, lamda, p=1):
    self.net.train()

    # Step 1: stdnorm-normalise raw images
    x = stdnorm(x, dims=[1, 2, 3])                                    # line 345

    # Step 2: create pos/neg channel-concatenated pairs
    x_pos, x_neg = get_pos_neg_batch_imgcats(x, x, p=p)               # line 346
    # x_pos shape: [B, 6, 32, 32]  (same image, cat'd with itself)
    # x_neg shape: [B, 6, 32, 32]  (image cat'd with a DIFFERENT image)

    # Step 3: forward pass through Layer 0
    out_pos = self.net(x_pos)                                          # line 348
    out_neg = self.net(x_neg)                                          # line 349

    # Step 4: compute goodness (mean squared ReLU activations)
    goodness_pos = self.net.relu(out_pos).pow(2).mean([1])             # line 351
    goodness_neg = self.net.relu(out_neg).pow(2).mean([1])             # line 352

    # Step 5: local SCFF loss + backward (gradient STAYS in Layer 0)
    self.optimizer.zero_grad()
    loss = (torch.log(1 + torch.exp(a * (-goodness_pos + threshold1))).mean([1, 2]).mean()
          + torch.log(1 + torch.exp(b * (goodness_neg  - threshold2))).mean([1, 2]).mean()
          + lamda * torch.norm(goodness_pos, p=2, dim=(1, 2)).mean())  # lines 356–358
    loss.backward()                                                    # line 359
    self.optimizer.step()                                              # line 360

    # Step 6: apply activation + pool, then CUT gradient graph with .detach()
    x_pos_pooled = self.pool(self.net.act(out_pos)).detach()           # line 363  ← SPLIT BOUNDARY
    x_neg_pooled = self.pool(self.net.act(out_neg)).detach()           # line 364  ← SPLIT BOUNDARY

    return x_pos_pooled, x_neg_pooled, gp, gn                         # line 369
```

These two detached tensors (`x_pos_pooled`, `x_neg_pooled`) are the **only thing the server ever receives from the client**.

### Server `train_step` — processes received activations

**`SCFF_CIFAR_Split.py` — Lines 417–461:**
```python
def train_step(self, x, x_neg, threshold1_list, threshold2_list, a, b, lamda_list,
               alleps, epoch, p=1):
    for i, net in enumerate(self.nets):
        net.train()

        if net.concat:
            # Re-create pos/neg pairs from the received activations
            x = stdnorm(x, dims=[1, 2, 3])                            # line 441
            x, x_neg = get_pos_neg_batch_imgcats(x, x, p=p)           # line 442

        out_pos = net(x)                                               # line 444
        out_neg = net(x_neg)                                           # line 445

        goodness_pos = net.relu(out_pos).pow(2).mean([1])              # line 447
        goodness_neg = net.relu(out_neg).pow(2).mean([1])              # line 448

        if epoch < alleps[i]:                                          # line 450 — epoch gating
            self.optimizers[i].zero_grad()
            loss = (torch.log(...).mean()
                  + torch.log(...).mean()
                  + lamda_list[i] * ...)                               # lines 452–454
            loss.backward()                                            # line 455
            # NO gradient goes back to client — detach broke the chain

        # Detach again before passing to next server layer
        x     = self.pools[i](net.act(out_pos)).detach()               # line 459
        x_neg = self.pools[i](net.act(out_neg)).detach()               # line 460
```

---

## 6. How Gradients Are Handled

### Complete Isolation at `.detach()`

```
Layer 0 loss  →  loss.backward()  →  gradients only in Layer 0 params
                                               │
                                         .detach()   ← line 363/364 (Split.py)
                                               │
Layer 1 loss  →  loss.backward()  →  gradients only in Layer 1 params
                                               │
                                         .detach()   ← line 459/460 (Split.py)
                                               │
Layer 2 loss  →  loss.backward()  →  gradients only in Layer 2 params
```

Each layer has its own `AdamW` optimizer. **No gradient tensor is ever transmitted** between client and server or between layers.

### Epoch Gating — layers freeze after `alleps` epochs

**`SCFF_CIFAR_Split.py` — Line 450 (server), Line 559 (trainer):**
```python
# Server side (line 450)
if epoch < alleps[i]:        # alleps defaults: [6, 6, 13] → Layer1 trains 6 ep, Layer2 13 ep
    self.optimizers[i].zero_grad()
    loss.backward()
    self.optimizers[i].step()
# After alleps[i], the layer passes activations forward but its weights are frozen

# Trainer side — client scheduler (line 559)
if epoch < self.alleps[0] and (nbbatches + 1) % self.period[0] == 0:
    self.client.scheduler.step()
```

---

## 7. Multi-Client Setup

**File:** `SCFF_CIFAR_MultiClient.py`

### Data Splitting

**`SCFF_CIFAR_MultiClient.py` — Lines 118–170 (`get_client_loaders`):**
```python
def get_client_loaders(num_clients, batchsize, iid=True):
    n_total = len(trainset)          # 50,000
    shard_size = n_total // num_clients

    if iid:
        indices = torch.randperm(n_total).tolist()   # random IID split
    else:
        targets = torch.tensor(trainset.targets)
        indices = targets.argsort().tolist()         # non-IID: sorted by class

    client_trainloaders = []
    for c in range(num_clients):
        start = c * shard_size
        end   = start + shard_size if c < num_clients - 1 else n_total
        client_subset = Subset(trainset, indices[start:end])
        loader = DataLoader(client_subset, batch_size=batchsize,
                            shuffle=True, num_workers=2)
        client_trainloaders.append(loader)
    ...
    return client_trainloaders, sup_trainloader, testloader
```

### Sequential Relay Training Loop

**`SCFF_CIFAR_MultiClient.py` — Lines 524–620 (`MultiClientTrainer.train`):**
```python
def train(self, total_rounds):
    for round_num in range(total_rounds):               # line 536 — one round ≈ 1 full epoch

        for client_id in range(self.num_clients):       # line 547 — sequential relay
            client_loader = self.client_loaders[client_id]

            for numbatch, (x, _) in enumerate(client_loader):
                x = x.to(device)

                # --- Client: Layer 0 trains on this shard's batch ---
                x_pos_pooled, x_neg_pooled, gp_c, gn_c = self.client.train_step(
                    x, self.threshold1[0], self.threshold2[0],
                    self.a, self.b, self.lamda[0], p=self.p
                )                                       # line 562–566

                # --- Server: Layers 1-2 train on received activations ---
                gp_s, gn_s = self.server.train_step(
                    x_pos_pooled, x_neg_pooled,         # ← detached tensors only
                    server_th1, server_th2,
                    self.a, self.b, server_lamda,
                    server_alleps, round_num, p=self.p
                )                                       # line 583–587

            # After this client's shard is done, weights relay to next client
            # (handled via get_weights / set_weights — see below)
```

### Does Raw Data Leave the Client?

**No. Never.**

The server only ever receives:
```python
x_pos_pooled  # shape [B, 96, H', W'] — detached activation maps
x_neg_pooled  # shape [B, 96, H', W'] — detached activation maps
```

The client **never transmits:**
- Raw pixel values
- Labels
- Layer 0 weights (those go to the *next client*, not the server)
- Any gradient

### The Weight Relay Mechanism

**`SCFF_CIFAR_MultiClient.py` — Lines 361–375 (`get_weights` / `set_weights`):**
```python
def get_weights(self):
    """Snapshot Layer 0 weights + optimizer state for handoff to next client."""
    return {
        'model':     copy.deepcopy(self.net.state_dict()),       # line 364
        'optimizer': copy.deepcopy(self.optimizer.state_dict()), # line 365
    }

def set_weights(self, state):
    """Receive relayed weights from previous client."""
    self.net.load_state_dict(state['model'])                     # line 370
    # Recreate optimizer bound to current net, then restore momentum/state
    self.optimizer = AdamW(self.net.parameters(), lr=self.lr,
                           weight_decay=self.weight_decay)
    self.optimizer.load_state_dict(state['optimizer'])           # line 373
    self.scheduler = ExponentialLR(self.optimizer, self.gamma)
```

The optimizer state carries **AdamW momentum buffers** — so learning doesn't cold-restart between clients. No gradient tensors are included; only learned weight values and adaptive learning rate state.

### How Gradients Flow in Multi-Client

```
Client 0 (shard 0):
  loss_layer0.backward() → grads update Layer 0 weights locally
  .detach() → activations sent to server (no grad)
  server: loss_layer1.backward(), loss_layer2.backward() — stay on server

  → get_weights() → deep copy of Layer 0 state_dict + optimizer state_dict

Client 1 (shard 1):
  set_weights(state) → loads Layer 0 weights from Client 0
  loss_layer0.backward() → grads update Layer 0 weights locally
  .detach() → activations sent to server
  server continues training Layers 1-2

  ... and so on for all N clients
```

**No gradient is ever transmitted anywhere.** Only weight values travel client→client, and only activation tensors travel client→server.

---

## 8. Evaluation & Linear Readout

After training all layers are frozen and features extracted from **all three layers simultaneously**.

**`SCFF_CIFAR_Split.py` — Lines 612–680 (`SplitTrainer.evaluate`):**
```python
def evaluate(self):
    # 1. Compute total feature dimension across layers in Layer_out=[0,1,2]
    lengths = calculate_output_length(self.Dims, all_nets, all_extra_pools,
                                      config.Layer_out, config.all_neurons)  # line ~620

    # 2. Build a tiny linear probe
    classifier = nn.Sequential(
        nn.Dropout(config.out_dropout),   # 0.2
        nn.Linear(lengths, 10)
    )                                                                  # line 273–280

    # 3. Readout training loop (50 epochs, Adam lr=0.001)
    for epoch_eval in range(50):
        for x, labels in suptrloader:
            with torch.no_grad():
                # Client extracts Layer 0 features
                x_after_client, client_outputs = self.client.extract_features(
                    x, config.dims_in, config.dims_out,
                    config.stdnorm_out, config.Layer_out)              # line 660

                # Server extracts Layers 1-2 features
                server_outputs = self.server.extract_features(
                    x_after_client, ...)                               # line 662

            # Concatenate all layer features → linear → cross-entropy
            outputs = torch.cat(client_outputs + server_outputs, dim=1)
            preds = classifier(outputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()                                           # line 671
```

The `extract_features` path for inference (no gradients):

**`SCFF_CIFAR_Split.py` — Lines 371–395 (`SplitClient.extract_features`):**
```python
def extract_features(self, x, dims_in, dims_out, stdnorm_out, Layer_out):
    self.net.eval()
    with torch.no_grad():
        x = stdnorm(x, dims=dims_in)                # line 384
        x = torch.cat((x, x), dim=1)               # cat with itself for concat=True layer
        x = self.pool(self.net.act(self.net(x)))    # forward + pool

        out = self.extra_pool(x)                    # extra spatial reduction
        if stdnorm_out:
            out = stdnorm(out, dims=dims_out)       # line 390
        out = out.flatten(start_dim=1)
        if 0 in Layer_out:
            outputs.append(out)                     # only collect if this layer is in readout set
    return x, outputs
```

---

## 9. How Accuracy Was Achieved

| Factor | Code Location | Detail |
|--------|--------------|--------|
| Local SCFF loss | Split.py line 356–358 | Each layer trains independently — split doesn't hurt accuracy |
| `.detach()` at split | Split.py line 363–364 | Mathematically identical to greedy layerwise training |
| Triangle activation | Split.py line 177–180 | `ReLU(x - mean(x))` — zero-centering improves feature quality |
| stdnorm per layer | Split.py line 147–152 | Spatial standardisation prevents scale drift between layers |
| Multi-layer readout | Split.py line 918 (`Layer_out=[2,1,0]`) | Features from all 3 layers → richer representation |
| ExponentialLR per layer | Split.py line 295–296 | Smooth independent LR decay per layer |
| AdamW weight decay | Split.py line 293 | L2 reg on weights |
| Relay with optimizer state | MultiClient.py line 364–365 | Momentum carries across client shards — no cold restart |

The split is **mathematically equivalent** to the non-split baseline because SCFF loss was already local per layer. The `.detach()` simply makes the layer boundary explicit.

---

## 10. Running the Scripts

### Single Client (Basic Split)

```bash
cd scff_split/

# Default run (NL=3, up to 13 epochs)
python SCFF_CIFAR_Split.py

# With explicit hyperparams
python SCFF_CIFAR_Split.py \
  --NL 3 \
  --lr 0.02 0.001 0.0004 \
  --th1 1 4 5 \
  --th2 2 5 7 \
  --alleps 6 6 13 \
  --device_num 0 \
  --save_model
```

### Multi-Client Split

```bash
# 10 clients IID (default)
python SCFF_CIFAR_MultiClient.py --num_clients 10

# 5 clients
python SCFF_CIFAR_MultiClient.py --num_clients 5

# Non-IID split (sorted by class — harder)
python SCFF_CIFAR_MultiClient.py --num_clients 10 --non_iid
```

### Output

Both scripts produce:
- Console: per-epoch goodness values + readout accuracy every 20 epochs + final train/test %
- PNG: `scff_split_accuracy.png` / `multiclient_{N}_accuracy.png`
- (Optional) Saved weights in `./results/` via `--save_model`

---

## 11. Config Reference

`config.json` → `CIFAR` section:

```json
{
  "CIFAR": {
    "layer_configs": [
      { "num":1, "ch_in":3,   "channels":96,   "kernel_size":5, "pooltype":"Max", "pad":2, ... },
      { "num":2, "ch_in":96,  "channels":384,  "kernel_size":3, "pooltype":"Max", "pad":1, ... },
      { "num":3, "ch_in":384, "channels":1536, "kernel_size":3, "pooltype":"Avg", "pad":1, ... }
    ],
    "opt_configs": [
      { "lr":0.01,   "weight_decay":0.0001, "gamma":0.7, "th1":0, "th2":1,  "epochs":6  },
      { "lr":0.002,  "weight_decay":0.0001, "gamma":0.8, "th1":5, "th2":9,  "epochs":4  },
      { "lr":0.0002, "weight_decay":0.0003, "gamma":1.0, "th1":6, "th2":10, "epochs":21 }
    ]
  }
}
```

**Hyperparameter reference:**

| Param | Meaning | Default |
|-------|---------|---------|
| `th1` | Goodness threshold — positive samples pushed above this | `[1, 4, 5]` |
| `th2` | Goodness threshold — negative samples pushed below this | `[2, 5, 7]` |
| `lamda` | L2 reg on positive goodness | `[0.0008, 0.0004, 0.0016]` |
| `gamma` | ExponentialLR decay factor (per scheduler step) | `[0.99, 0.9, 0.99]` |
| `period` | Scheduler step every N batches | `[500, 500, 500]` |
| `alleps` | Epochs each layer trains for (then freezes) | `[6, 6, 13]` |
| `concat` | Whether layer uses pos/neg channel-cat mechanism | `(1, 0, 1)` |
| `act` | Activation: `"triangle"` (zero-centred ReLU) or `"relu"` | per-layer |

---

*README generated from `SCFF_CIFAR_Split.py` and `SCFF_CIFAR_MultiClient.py`.*
# SCFF-with-split-learning
