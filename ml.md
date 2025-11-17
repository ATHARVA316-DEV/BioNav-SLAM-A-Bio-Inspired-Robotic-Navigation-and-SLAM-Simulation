# ML Integration Plan for BioNav-SLAM

**Goal:** refactor the project so you can *easily add ML models* (training + inference) that learn from simulation data (e.g., place recognition, odometry correction, loop-closure classifier, policy networks). Provide a clear pipeline: **collect data → train offline → load model for inference in simulation** with reproducible configs and tests.

---

## High-level design

1. Keep the simulation core (physics, Central Complex modules, planner) separate and deterministic.
2. Add `bionav.ml` as the ML layer: dataset creation utilities, trainer abstraction, model zoo, and an inference wrapper.
3. Provide a `data/` folder for logged episodes (CSV / NumPy / TFRecords / Torch files).
4. Provide CLI flags so you can run simulation in `--collect-data`, `--train`, or `--infer` modes.
5. Provide standardized model interface with methods: `fit(dataset, cfg)`, `save(path)`, `load(path)`, `predict(obs)`.

---

## Proposed repo layout

```
BioNav-SLAM/
├─ bionav/
│  ├─ __init__.py
│  ├─ core.py          # Central Complex math, TB1, CPU4 etc.
│  ├─ sim.py           # Env + run_simulation()
│  ├─ planner.py      # A* / path planning
│  ├─ gui.py           # visualization (thin wrapper)
│  ├─ ml.py            # high-level ML trainer & model interface
│  ├─ datasets.py      # dataset builder & loaders
│  ├─ io.py            # save/load traces/configs
│  └─ utils.py         # helpers
├─ configs/
│  ├─ experiment.yaml
├─ data/
│  └─ episodes/       # .npz or .npy logged runs
├─ models/
│  └─ saved_models/
├─ tests/
├─ bio_nav.py         # CLI entrypoint
├─ requirements.txt
└─ README.md
```

---

## requirements.txt (additions for ML)

```
numpy>=1.24
matplotlib>=3.6
pytest>=7.0
pillow>=9.0
scikit-learn>=1.2
torch>=2.0    # or replace with tensorflow if you prefer
tqdm
yaml
```

---

## Key design patterns & interfaces

### 1) Data collection (simulation → disk)

* `run_simulation(..., collect=True, out_dir=...)` records per-step observations and actions into an episode file (prefer `.npz` containing arrays: `observations`, `actions`, `rewards`, `meta`).
* Observations are structured dictionaries turned into arrays using `datasets.py:serialize_observation(obs)`.

**Why:** offline training is easier and deterministic.

### 2) Dataset & loader

* `bionav/datasets.py` exposes `EpisodeDataset` (PyTorch `Dataset`) that loads `.npz` files and yields `(x, y)` pairs for training.

### 3) Trainer & model abstraction

* `bionav/ml.py` exposes a simple `Trainer` class with `.train(dataset, cfg)` and `.evaluate()` plus model interface wrapper classes (e.g., `TorchModelWrapper`) that implement `save/load/predict`.

### 4) Inference in-sim

* Provide `bionav/ml_inference.py` or keep a `predict_fn = model.predict` reference; at each sim step call `model.predict(observation)` and use output to modify agent behavior (e.g., correct odometry, give loop-closure signal, or output action probabilities).

---

## Example files (copy-paste ready)

### `bionav/datasets.py` (PyTorch dataset loader)

```python
# bionav/datasets.py
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

class EpisodeDataset(Dataset):
    """Loads .npz episode files saved by run_simulation(collect=True).
    Each .npz contains: observations (N x D), actions (N x A), meta dict.
    """
    def __init__(self, folder, transform=None):
        self.folder = Path(folder)
        self.files = list(self.folder.glob("*.npz"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        obs = data['observations']
        acts = data['actions']
        # default: return full time-series; training code may slice subsequences
        sample = {'observations': obs.astype('float32'), 'actions': acts.astype('float32')}
        if self.transform:
            sample = self.transform(sample)
        return sample
```

---

### `bionav/ml.py` (trainer + simple model wrapper)

```python
# bionav/ml.py
import torch
from torch import nn
from torch.utils.data import DataLoader
import os

class TorchModelWrapper:
    def __init__(self, model: nn.Module, device='cpu'):
        self.model = model.to(device)
        self.device = device

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def predict(self, x_np):
        # x_np: numpy array, shape (B, D) or (D,)
        import numpy as np
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(x_np)).float().to(self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            out = self.model(x)
            return out.cpu().numpy()

class Trainer:
    def __init__(self, model_wrapper: TorchModelWrapper, lr=1e-3, device='cpu'):
        self.w = model_wrapper
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.w.model.parameters(), lr=lr)

    def train(self, dataset, epochs=10, batch_size=64):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for ep in range(epochs):
            running = 0.0
            for batch in loader:
                obs = batch['observations']  # shape B x T x D OR B x D
                acts = batch['actions']
                # For simplicity assume B x D -> regression
                obs = obs.float().to(self.device)
                acts = acts.float().to(self.device)
                preds = self.w.model(obs)
                loss = self.criterion(preds, acts)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running += loss.item()
            print(f"Epoch {ep}: loss={running/len(loader):.6f}")
```

> Note: This trainer assumes observations/actions are already preprocessed into fixed-size arrays. For sequence models (RNNs/Transformers), modify dataset to return subsequences.

---

### Example simple MLP model (bionav/models/simple_mlp.py)

```python
# bionav/models/simple_mlp.py
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=[128,64]):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())
```

---

### Integration: collect data from sim (`sim.py` changes)

* Add a `collect_episode` flag and a helper `save_episode(path, obs_list, actions_list, meta)` that writes `.npz` files.
* Per-step, record the observation structure (e.g., `[x,y,theta,sensor_readings,...]`) and action taken.

**Suggested naming:** `data/episodes/expname_seed_0001.npz`

---

### Integration: run inference in sim

* After training, load model via `TorchModelWrapper.load(path)` from `models/saved_models/model.pth`
* Pass the `model.predict(obs_vector)` output to the decision layer. For example, a trained model can produce a corrected `dx, dy` to fuse with CPU4.

---

## Example CLI modes (bio_nav.py)

```
--mode {run,collect,train,infer}
--collect-out data/episodes/
--model-out models/saved_models/
--config configs/experiment.yaml
```

* `collect`: runs N episodes saving `.npz` files.
* `train`: loads dataset and trains the model.
* `infer`: starts sim and loads model for online inference.

---

## Unit testing & CI

* Add tests that verify dataset loader loads arrays with expected shapes
* Add a smoke test that creates a tiny model, trains for 1 epoch on a generated dataset, saves and reloads it, and runs `predict`.

---

## Example quick-run workflow

1. `python bio_nav.py --mode collect --episodes 50 --out data/episodes/`  # gather data
2. `python bio_nav.py --mode train --config configs/experiment.yaml`       # trains model
3. `python bio_nav.py --mode infer --model models/saved_models/latest.pth` # run sim with ML

---

## Next steps I can do for you right now (pick any):

* Generate the exact `bionav/datasets.py`, `bionav/ml.py`, `bionav/models/simple_mlp.py` files as ready-to-paste code (I already included them here).
* Create `bio_nav.py` CLI wrapper code that wires `--mode` and the three workflows.
* Create a small example `collect -> train -> infer` demo using synthetic data so you can run end-to-end.
* Make a PR-style diff or zip with the new files.


