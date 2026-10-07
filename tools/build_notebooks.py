"""Build the course notebooks. Run: python build_notebooks.py <repo>/notebooks"""

import sys
from pathlib import Path

import nbformat as nbf

SETUP = """import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from minihep.utils import ensure_toy_data, repo_root

ROOT = repo_root()
os.chdir(ROOT)  # every path below is relative to the repository root
sys.path.insert(0, str(ROOT))  # so `import exercises...` works from a notebook
FAST = bool(os.environ.get("MINIHEP_FAST"))  # smaller numbers for a quick check that everything runs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(min(8, os.cpu_count() or 1))
print("device:", DEVICE, "| torch", torch.__version__, "| fast mode" if FAST else "")"""


def nb(cells):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    book.cells = [nbf.v4.new_markdown_cell(src.strip()) if kind == "md" else nbf.v4.new_code_cell(src.strip()) for kind, src in cells]
    return book


NOTEBOOKS = {}

# ----------------------------------------------------------------------------------------------- week 1
NOTEBOOKS["week01_reading_hepattn.ipynb"] = nb(
    [
        (
            "md",
            """
# Week 1: reading a codebase like hepattn

This notebook is a tour of the Python features from this week's lesson, looked at *from the outside*
with `inspect`. You will use these same tools on hepattn itself: when a config says
`class_path: hepattn.models.Encoder`, `inspect.signature` tells you every option it accepts.

Run each cell with **Shift+Enter**. Cells marked **Your turn** are for you to edit.
""",
        ),
        ("code", SETUP),
        (
            "md",
            """
## 1. Signatures are the documentation (and the config schema)

`inspect.signature` shows a function's parameters, defaults and type hints. LightningCLI reads exactly
this to decide what a YAML block may contain (week 7).
""",
        ),
        (
            "code",
            """import inspect

from minihep.models.encoder import Encoder, EncoderLayer

print("Encoder", inspect.signature(Encoder.__init__))
print("EncoderLayer", inspect.signature(EncoderLayer.__init__))""",
        ),
        (
            "md",
            """
Notice `**layer_kwargs` on `Encoder`: anything it does not recognise is forwarded to every `EncoderLayer`.
That is how `hybrid_norm: true` in a hepattn encoder config reaches the layers.

## 2. Registries: strings to classes
""",
        ),
        (
            "code",
            """from minihep.models.norm import NORM_TYPES

for name, cls in NORM_TYPES.items():
    print(f"{name:14s} -> {cls.__module__}.{cls.__qualname__}")

norm = NORM_TYPES["RMSNorm"](8)  # what Residual(..., norm="RMSNorm") does internally
print(norm)""",
        ),
        (
            "md",
            """
## 3. Closures carry state

A closure stores the variables it captured in `__closure__`. This is how a FlexAttention `mask_mod`
remembers its window size.
""",
        ),
        (
            "code",
            """from exercises.week01.ex02_closures import make_window_mask, make_wrapped_window_mask

try:
    mask = make_window_mask(4)
    print("captured:", [c.cell_contents for c in mask.__closure__])
    print("mask(0, 2) =", mask(0, 2), "| mask(0, 3) =", mask(0, 3))
except NotImplementedError as err:
    print("Finish exercise 2 first:", err)""",
        ),
        (
            "code",
            """# The late-binding trap, live
fs = [lambda x: i * x for i in range(4)]
print([f(10) for f in fs], "<- every lambda sees the final i")""",
        ),
        (
            "md",
            """
## 4. The dict-key convention

Every tensor in hepattn travels in a flat dict. Keys are built with f-strings from an object name and a
field. Learn to read them as `{object}_{field}`.
""",
        ),
        (
            "code",
            """keys = ["hit_x", "hit_valid", "hit_embed", "particle_valid", "particle_hit_valid", "track_hit_logit", "key_is_hit"]
for key in keys:
    head, _, rest = key.partition("_")
    print(f"{key:20s} object={head:9s} field={rest}")""",
        ),
        (
            "md",
            """
## Your turn

1. Clone hepattn on your laptop (week 1 lesson, section "Setting up") and find every registry:
   `git grep -n "_TYPES\\b = \\|_fns = {\\|SOLVERS = {" src/hepattn`.
2. Run `inspect.signature` on `hepattn.models.MaskFormer.__init__` (inside `pixi shell` in the hepattn repo)
   and match each parameter to a key in `src/hepattn/experiments/trackml/configs/tracking-eta4-pt600-flex.yaml`.
3. Find the line in `hepattn/models/encoder.py` that writes `self.seq_len[0] = seq_len` and explain, in a
   comment cell below, why it cannot be `self.seq_len = torch.tensor([seq_len])`.
""",
        ),
    ]
)

# ----------------------------------------------------------------------------------------------- week 2
NOTEBOOKS["week02_toy_detector.ipynb"] = nb(
    [
        (
            "md",
            """
# Week 2: the toy detector and the data pipeline

We generate the dataset used for the rest of the course, look at events, and check that the
`ToyTrackingDataset` you wrote produces sensible tensors.
""",
        ),
        ("code", SETUP),
        (
            "code",
            """data_dir = ensure_toy_data(num_train=200 if FAST else 2000, num_val=50 if FAST else 200, num_test=50 if FAST else 200)
print(data_dir, {s: len(list((data_dir / s).glob('*-hits.parquet'))) for s in ('train', 'val', 'test')})""",
        ),
        (
            "md",
            """
## 1. One raw event

Every hit has a `particle_id`; 0 means noise. Colours below are particles. Look at the transverse view:
low-pT particles curve more (radius `R = pT / (0.3 B)`). In the r-z view tracks are straight lines.
""",
        ),
        (
            "code",
            """import pandas as pd

from minihep.plotting import event_display

hits = pd.read_parquet(data_dir / "train" / "event000000001-hits.parquet")
parts = pd.read_parquet(data_dir / "train" / "event000000001-parts.parquet")
print(hits.head(), "\\n", parts.head())
event_display(hits.x / 1000, hits.y / 1000, hits.z / 1000, hits.particle_id, title="event 1 (all particles)")
plt.show()""",
        ),
        (
            "code",
            """parts_all = pd.concat([pd.read_parquet(p) for p in sorted((data_dir / "train").glob("*-parts.parquet"))[:200]])
pt = np.hypot(parts_all.px, parts_all.py)
eta = np.arcsinh(parts_all.pz / pt)
fig, axes = plt.subplots(1, 4, figsize=(18, 3.5))
axes[0].hist(pt, bins=60, range=(0, 10)); axes[0].set_xlabel("pT [GeV]"); axes[0].set_yscale("log")
axes[1].hist(eta, bins=60); axes[1].set_xlabel("eta")
axes[2].hist(parts_all.nhits, bins=np.arange(10) - 0.5); axes[2].set_xlabel("hits per particle")
axes[3].hist([len(pd.read_parquet(p)) for p in sorted((data_dir / "train").glob("*-hits.parquet"))[:200]], bins=30)
axes[3].set_xlabel("hits per event")
plt.tight_layout(); plt.show()""",
        ),
        (
            "md",
            """
## 2. Through your Dataset

`dataset[i]` returns `(inputs, targets)`. Check every shape against the docstring of `minihep/data/dataset.py`.
""",
        ),
        (
            "code",
            """from minihep.data.dataset import ToyTrackingDataset

inputs_cfg = {"hit": ["x", "y", "z", "r", "eta", "phi"]}
targets_cfg = {"hit": ["on_valid_particle", "is_first"], "particle": ["pt", "eta", "phi"]}
ds = ToyTrackingDataset(str(data_dir / "train"), inputs_cfg, targets_cfg, event_max_num_particles=32)
inputs, targets = ds[0]
for name, d in (("inputs", inputs), ("targets", targets)):
    for k, v in d.items():
        print(f"{name:8s} {k:24s} {tuple(v.shape)!s:14s} {v.dtype}")""",
        ),
        (
            "code",
            """fig, axes = plt.subplots(1, 3, figsize=(20, 5))
order = torch.argsort(inputs["hit_phi"][0])
axes[0].imshow(targets["particle_hit_valid"][0][:, order], aspect="auto", interpolation="nearest", cmap="Greys")
axes[0].set_xlabel("hit (sorted by phi)"); axes[0].set_ylabel("particle slot"); axes[0].set_title("particle_hit_valid: the mask targets")
hits_v, parts_v = ds.load_event(0)
labels = np.where(hits_v.on_valid_particle, hits_v.particle_id, 0)
event_display(hits_v.x, hits_v.y, hits_v.z, labels, title="reconstructable particles only (pT > 1 GeV)", axes=axes[1:])
plt.tight_layout(); plt.show()""",
        ),
        (
            "md",
            """
## 3. Parquet vs HDF5

Pack the training split into one HDF5 file and compare size and read speed.
""",
        ),
        (
            "code",
            """import time

import h5py

from minihep.data.prep import parquet_dir_to_hdf5

h5_path = data_dir / "train.h5"
if not h5_path.exists():
    parquet_dir_to_hdf5(data_dir / "train", h5_path)
size_pq = sum(p.stat().st_size for p in (data_dir / "train").glob("*.parquet")) / 1e6
print(f"parquet: {size_pq:.1f} MB   hdf5 (lzf): {h5_path.stat().st_size / 1e6:.1f} MB")
with h5py.File(h5_path) as f:
    name = sorted(f)[0]
    f[name].visititems(lambda path, obj: print(f"  {name}/{path}", getattr(obj, "shape", ""), getattr(obj, "dtype", "")))

for source in (str(data_dir / "train"), str(h5_path)):
    d = ToyTrackingDataset(source, inputs_cfg, targets_cfg)
    t0 = time.perf_counter(); [d[i] for i in range(min(100, len(d)))]
    print(f"{Path(source).name:10s} {(time.perf_counter() - t0) * 10:.2f} ms/event")""",
        ),
        (
            "md",
            """
## Your turn

1. Change `particle_min_pt` to 0.5 and see how the number of target particles per event changes.
2. Make `pad_collate` batch 4 events and `imshow` the padded `hit_valid` mask.
3. In hepattn, open `experiments/trackml/data.py` and find the line that casts inputs with `.half()`.
   Which TrackML input in your vault notes overflowed because of it? (Week 8 returns to this.)
""",
        ),
    ]
)

# ----------------------------------------------------------------------------------------------- week 3
NOTEBOOKS["week03_mlp_hit_filter.ipynb"] = nb(
    [
        (
            "md",
            """
# Week 3: training a hit filter by hand

Train the MLP hit filter with *your* training loop, look at the curves, and choose a working point.
""",
        ),
        ("code", SETUP),
        (
            "code",
            """from torch import nn

from exercises.week03.ex02_training_loop import LRConfig, evaluate, train
from minihep.data.dataset import ToyTrackingDataset
from minihep.models.dense import Dense
from minihep.models.hitfilter import HitFilter
from minihep.models.input import InputNet
from minihep.models.tasks import HitFilterTask

data_dir = ensure_toy_data(num_train=200 if FAST else 2000, num_val=50 if FAST else 200)
fields = ["x", "y", "z", "r", "eta", "phi"]
cfg = {"inputs": {"hit": fields}, "targets": {"hit": ["on_valid_particle"]}}
train_ds = ToyTrackingDataset(str(data_dir / "train"), **cfg)
val_ds = ToyTrackingDataset(str(data_dir / "val"), **cfg)

def make_mlp(dim=64):
    net = InputNet("hit", Dense(len(fields), dim, hidden_layers=[dim, dim]), fields)
    return HitFilter(nn.ModuleList([net]), nn.ModuleList([HitFilterTask("hit_filter", "hit", "on_valid_particle", dim)]))

model = make_mlp()
print(model)
print(sum(p.numel() for p in model.parameters()), "parameters")""",
        ),
        (
            "code",
            """steps = 300 if FAST else 3000
history = train(model, train_ds, num_steps=steps, cfg=LRConfig(initial=1e-4, max=2e-3, end=1e-5, pct_start=0.05), device=DEVICE)
fig, axes = plt.subplots(1, 3, figsize=(16, 3.5))
smooth = np.convolve(history.loss, np.ones(50) / 50, mode="valid")
axes[0].plot(history.loss, alpha=0.3); axes[0].plot(smooth); axes[0].set_title("loss")
axes[1].plot(history.lr); axes[1].set_title("learning rate (OneCycleLR)")
axes[2].plot(history.grad_norm, alpha=0.5); axes[2].set_yscale("log"); axes[2].set_title("gradient norm (before clipping)")
plt.show()
print(evaluate(model, val_ds, "hit_filter", device=DEVICE))""",
        ),
        (
            "md",
            """
## Choosing a threshold

The filter's job is to throw away noise and uninteresting hits *without losing hits the tracker needs*.
So we want very high recall, then as much precision as possible. hepattn's TrackML filter uses a
threshold of 0.1 (600 MeV) or 0.314 (900 MeV), chosen exactly this way.
""",
        ),
        (
            "code",
            """@torch.no_grad()
def collect(model, ds):
    model.eval()
    probs, truth = [], []
    for i in range(len(ds)):
        inputs, targets = ds[i]
        out = model({k: v.to(DEVICE) for k, v in inputs.items()})
        probs.append(out["final"]["hit_filter"]["hit_logit"].sigmoid().cpu()[0])
        truth.append(targets["hit_on_valid_particle"][0])
    model.train()
    return torch.cat(probs), torch.cat(truth)

probs, truth = collect(model, val_ds)
thresholds = np.linspace(0.01, 0.99, 99)
recall = [((probs >= t) & truth).sum().item() / truth.sum().item() for t in thresholds]
kept = [(probs >= t).float().mean().item() for t in thresholds]
plt.plot(thresholds, recall, label="recall (fraction of good hits kept)")
plt.plot(thresholds, kept, label="fraction of all hits kept")
plt.xlabel("threshold"); plt.legend(); plt.grid(alpha=0.3); plt.show()""",
        ),
        (
            "md",
            """
## Your turn

1. Why can an MLP that sees one hit at a time not tell a 0.9 GeV particle from a 1.1 GeV one? Which
   information would it need? (Next week's transformer gets it.)
2. Retrain with `loss_fn="focal"` in the task and compare the threshold plot.
3. Set `grad_clip=None` and a 10x higher max learning rate. What happens to the gradient norm plot?
""",
        ),
    ]
)

# ----------------------------------------------------------------------------------------------- week 4
NOTEBOOKS["week04_transformer_filter.ipynb"] = nb(
    [
        (
            "md",
            """
# Week 4: a transformer hit filter

Same task as last week, but every hit can now attend to every other hit. Compare with the MLP, then look
inside the attention.
""",
        ),
        ("code", SETUP),
        (
            "code",
            """from torch import nn

from exercises.week03.ex02_training_loop import LRConfig, evaluate, train
from minihep.data.dataset import ToyTrackingDataset
from minihep.models.dense import Dense
from minihep.models.encoder import Encoder
from minihep.models.hitfilter import HitFilter
from minihep.models.input import InputNet
from minihep.models.posenc import PositionEncoder
from minihep.models.tasks import HitFilterTask

data_dir = ensure_toy_data(num_train=200 if FAST else 2000, num_val=50 if FAST else 200)
fields = ["x", "y", "z", "r", "eta", "phi"]
cfg = {"inputs": {"hit": fields}, "targets": {"hit": ["on_valid_particle"]}}
train_ds = ToyTrackingDataset(str(data_dir / "train"), **cfg)
val_ds = ToyTrackingDataset(str(data_dir / "val"), **cfg)

def make_filter(encoder: bool, dim=64):
    posenc = PositionEncoder("hit", ["r", "eta", "phi"], dim, sym_fields=["phi"], alpha=10)
    net = InputNet("hit", Dense(len(fields), dim, hidden_layers=[dim]), fields, posenc=posenc)
    enc = Encoder(num_layers=3, dim=dim, attn_kwargs={"num_heads": 4}, norm="LayerNorm") if encoder else None
    task = HitFilterTask("hit_filter", "hit", "on_valid_particle", dim)
    return HitFilter(nn.ModuleList([net]), nn.ModuleList([task]), encoder=enc, input_sort_field="phi")

steps = 300 if FAST else 3000
lr = LRConfig(initial=1e-4, max=1e-3, end=1e-5, pct_start=0.05)
results = {}
for label, use_encoder in (("MLP", False), ("transformer", True)):
    torch.manual_seed(0)
    model = make_filter(use_encoder)
    hist = train(model, train_ds, steps, lr, device=DEVICE)
    results[label] = (model, hist, evaluate(model, val_ds, "hit_filter", device=DEVICE))
    print(label, results[label][2])""",
        ),
        (
            "code",
            """for label, (_, hist, _) in results.items():
    plt.plot(np.convolve(hist.loss, np.ones(50) / 50, mode="valid"), label=label)
plt.yscale("log"); plt.xlabel("step"); plt.ylabel("loss (smoothed)"); plt.legend(); plt.show()""",
        ),
        (
            "md",
            """
## Looking inside: who does a hit attend to?

`F.scaled_dot_product_attention` never hands back the attention weights (that is the point of a fused
kernel), so recompute them for the first layer with the projections and your `naive_attention`.
""",
        ),
        (
            "code",
            """from exercises.week04.ex01_attention_from_scratch import naive_attention

model = results["transformer"][0].cpu().eval()
inputs, targets = val_ds[0]
with torch.no_grad():
    x = model.input_nets[0](inputs)
    layer = model.encoder.layers[0]
    attn = layer.attn.fn
    xn = layer.attn.norm(x)
    q, k, v = (attn.separate_heads(t) for t in attn.project_qkv(xn, xn, xn))
    _, weights = naive_attention(q, k, v)  # (1, H, N, N)

hits, _ = val_ds.load_event(0)
query = int(np.flatnonzero(hits.on_valid_particle.to_numpy())[0])
w = weights[0, :, query].mean(0).numpy()
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(hits.x, hits.y, c=w, s=8 + 400 * w / w.max(), cmap="viridis")
same = hits.particle_id == hits.particle_id.iloc[query]
ax.scatter(hits.x[same], hits.y[same], facecolors="none", edgecolors="red", s=80, label="same particle")
ax.scatter(hits.x.iloc[query], hits.y.iloc[query], marker="*", c="red", s=300, label="query hit")
ax.set_aspect("equal"); ax.legend(); ax.set_title("layer-0 attention of one hit, averaged over heads"); plt.show()""",
        ),
        (
            "md",
            """
## Your turn

1. Remove the positional encoding (`posenc=None`). Does the transformer still beat the MLP? Why might it,
   given the raw x, y, z are still inputs?
2. Try `hybrid_norm=True` in the Encoder and `norm="RMSNorm"`. Any difference at this size?
3. Count parameters of both models. Is the comparison fair?
""",
        ),
    ]
)

# ----------------------------------------------------------------------------------------------- week 5
NOTEBOOKS["week05_efficient_attention.ipynb"] = nb(
    [
        (
            "md",
            """
# Week 5: windows, block masks and kernels

Three experiments: what the masks look like, whether sorting makes windows work, and what each attention
kernel costs on *your* GPU.
""",
        ),
        ("code", SETUP),
        (
            "code",
            """from torch.nn.attention.flex_attention import create_block_mask

from minihep.flex.masks import mask_mod_to_dense, sliding_window_mask, sliding_window_mask_wrapped, strided_window_mask

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].imshow(mask_mod_to_dense(sliding_window_mask(16), 64, 64), cmap="Greys"); axes[0].set_title("sliding window (encoder)")
axes[1].imshow(mask_mod_to_dense(sliding_window_mask_wrapped(16, torch.tensor([64])), 64, 64), cmap="Greys"); axes[1].set_title("wrapped (phi is periodic)")
axes[2].imshow(mask_mod_to_dense(strided_window_mask(16, 16, 64), 16, 64), cmap="Greys", aspect="auto"); axes[2].set_title("strided local cross-attention (decoder)")
for ax in axes: ax.set_xlabel("key"); ax.set_ylabel("query")
plt.show()
print(create_block_mask(sliding_window_mask(512), B=None, H=None, Q_LEN=4096, KV_LEN=4096, device="cpu"))""",
        ),
        (
            "md",
            """
## Does sorting make windows useful?

Pair recall: the fraction of same-particle hit pairs that end up inside each other's window.
""",
        ),
        (
            "code",
            """from exercises.week05.ex03_locality import cell_sort_value, pair_recall, positions_from_sort_value
from minihep.data.dataset import ToyTrackingDataset

data_dir = ensure_toy_data(num_train=50 if FAST else 2000)
ds = ToyTrackingDataset(str(data_dir / "train"), {"hit": ["x"]}, {})
rng = np.random.default_rng(0)
windows = [4, 8, 16, 32, 64, 128]
curves = {"random": [], "phi": [], "eta-phi cells": []}
for w in windows:
    rec = {k: [] for k in curves}
    for i in range(10):
        hits, _ = ds.load_event(i)
        pid, n = hits.particle_id.to_numpy(), len(hits)
        rec["random"].append(pair_recall(pid, rng.permutation(n), w, n))
        rec["phi"].append(pair_recall(pid, positions_from_sort_value(hits.phi.to_numpy()), w, n))
        rec["eta-phi cells"].append(pair_recall(pid, positions_from_sort_value(cell_sort_value(hits.eta.to_numpy(), hits.phi.to_numpy(), 6)), w, n))
    for k in curves: curves[k].append(np.mean(rec[k]))
for k, v in curves.items(): plt.plot(windows, v, "o-", label=k)
plt.xscale("log", base=2); plt.xlabel("window size"); plt.ylabel("same-particle pair recall"); plt.legend(); plt.grid(alpha=0.3); plt.show()""",
        ),
        (
            "md",
            """
## Kernel benchmark

Time and peak memory of one attention call versus sequence length. On CPU only the small sizes run.
On your GPU, compare full SDPA, SDPA with a dense window mask, compiled flex with a window BlockMask and
FlashAttention-2 with a window (if you installed the `fa2` environment: `pixi run -e fa2 lab`).
""",
        ),
        (
            "code",
            """import time

import pandas as pd
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_mask, flex_attention


def time_fn(fn, *args, warmup=3, iters=10, device="cpu"):
    # median milliseconds per call, synchronising the GPU around each call (week 8 explains why)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(iters):
        sync(); start = time.perf_counter(); fn(*args); sync()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

dtype = torch.float16 if DEVICE == "cuda" else torch.float32
flex = torch.compile(flex_attention, dynamic=False) if DEVICE == "cuda" else flex_attention
window, heads, dim = 256, 8, 32
sizes = [512, 1024, 2048] if (FAST or DEVICE == "cpu") else [1024, 2048, 4096, 8192, 16384, 32768]
rows = []
for n in sizes:
    q = torch.randn(1, heads, n, dim, device=DEVICE, dtype=dtype)
    mod = sliding_window_mask(window)
    bm = create_block_mask(mod, B=None, H=None, Q_LEN=n, KV_LEN=n, device=DEVICE)
    row = {"N": n, "sdpa full": time_fn(F.scaled_dot_product_attention, q, q, q, device=DEVICE)}
    if n <= 16384:
        dense = create_mask(mod, 1, 1, n, n, device=DEVICE)
        row["sdpa dense mask"] = time_fn(lambda: F.scaled_dot_product_attention(q, q, q, attn_mask=dense), device=DEVICE)
    row["flex window"] = time_fn(lambda: flex(q, q, q, block_mask=bm), device=DEVICE)
    if flash_attn_func is not None and DEVICE == "cuda":
        qf = q.transpose(1, 2).contiguous()
        row["flash window"] = time_fn(lambda: flash_attn_func(qf, qf, qf, window_size=(window // 2, window // 2)), device=DEVICE)
    rows.append(row)
table = pd.DataFrame(rows).set_index("N")
print(table.round(3))
table.plot(marker="o", logx=True, logy=True, ylabel="ms per call"); plt.show()""",
        ),
        (
            "md",
            """
## Your turn

1. In the table, where does windowed attention start to beat full attention? A dense mask still makes SDPA
   visit every (query, key) pair, so what speed-up should you expect from it, and does your table agree?
   (Run on an otherwise idle GPU, and close other notebooks first.)
2. Change `window` to 64 and 1024. How does flex's time scale with the window at fixed N?
3. Read `per_head_window_mask_mod` in hepattn's `flex/per_head_window.py`. What does it capture in its
   closure that `sliding_window_mask` does not, and why must it be rebuilt for every event?
""",
        ),
    ]
)

# ----------------------------------------------------------------------------------------------- week 6
NOTEBOOKS["week06_maskformer.ipynb"] = nb(
    [
        (
            "md",
            """
# Week 6: MaskFormer tracking

Train your tracker, watch Hungarian matching at work, and measure tracking efficiency.
""",
        ),
        ("code", SETUP),
        (
            "code",
            """from exercises.week03.ex02_training_loop import LRConfig, train
from exercises.week06.ex02_build_tracker import build_tracker
from minihep.data.dataset import ToyTrackingDataset
from minihep.metrics import double_majority

data_dir = ensure_toy_data(num_train=200 if FAST else 2000, num_val=50 if FAST else 200)
fields = ["x", "y", "z", "r", "eta", "phi"]
cfg = {"inputs": {"hit": fields}, "targets": {"hit": ["on_valid_particle"], "particle": ["pt", "eta", "phi"]}, "event_max_num_particles": 32}
train_ds = ToyTrackingDataset(str(data_dir / "train"), **cfg)
val_ds = ToyTrackingDataset(str(data_dir / "val"), **cfg)
torch.manual_seed(0)
model = build_tracker(fields, dim=64, num_queries=32, num_encoder_layers=3, num_decoder_layers=3)
print(sum(p.numel() for p in model.parameters()), "parameters")""",
        ),
        (
            "md",
            """
## The cost matrix before training

Rows are predictions (queries), columns targets. The matcher picks one cell per column (red dots).
""",
        ),
        (
            "code",
            """def show_matching(model, ds, i=0):
    model.eval()
    inputs, targets = ds[i]
    with torch.no_grad():
        out = model({k: v.to(DEVICE) for k, v in inputs.items()})
        t = {k: v.to(DEVICE) for k, v in targets.items()}
        final = out["final"]
        cost = sum(c for task in model.tasks for c in task.cost(final[task.name], t).values())[0].cpu()
        idx = model.matcher(cost[None], targets["particle_valid"])[0]
    n_true = int(targets["particle_valid"].sum())
    plt.imshow(cost[:, :n_true], aspect="auto", cmap="viridis"); plt.colorbar(label="cost")
    plt.scatter(range(n_true), idx[:n_true], c="red", s=15)
    plt.xlabel("target particle"); plt.ylabel("query"); plt.show()
    model.train()

model.to(DEVICE)
show_matching(model, val_ds)""",
        ),
        (
            "code",
            """steps = 300 if FAST else 6000
history = train(model, train_ds, steps, LRConfig(initial=1e-4, max=1e-3, end=1e-5, pct_start=0.05), device=DEVICE)
plt.plot(np.convolve(history.loss, np.ones(50) / 50, mode="valid")); plt.yscale("log"); plt.title("training loss"); plt.show()
show_matching(model, val_ds)""",
        ),
        (
            "md",
            """
## Efficiency, fakes and duplicates on the validation set
""",
        ),
        (
            "code",
            """import pandas as pd


@torch.no_grad()
def evaluate_tracking(model, ds):
    model.eval()
    rows, per_particle = [], []
    for i in range(len(ds)):
        inputs, targets = ds[i]
        out = model({k: v.to(DEVICE) for k, v in inputs.items()})
        preds = model.predict(out)["final"]
        pv, pm = preds["track_valid"]["track_valid"][0].cpu(), preds["track_hit_valid"]["track_hit_valid"][0].cpu()
        tv, tm = targets["particle_valid"][0], targets["particle_hit_valid"][0]
        rows.append(double_majority(pv, pm, tv, tm))
        # per-particle efficiency for an efficiency-vs-pT plot
        shared = pm[pv].long() @ tm.long().T
        n_track = pm[pv].sum(-1, keepdim=True).clamp_min(1)
        ok = ((shared / n_track > 0.5) & (shared / tm.sum(-1).clamp_min(1) > 0.5)).any(0)
        for p in range(int(tv.sum())):
            per_particle.append((targets["particle_pt"][0, p].item(), bool(ok[p])))
    model.train()
    return pd.DataFrame(rows).mean(), pd.DataFrame(per_particle, columns=["pt", "found"])


summary, per_particle = evaluate_tracking(model, val_ds)
print(summary)
bins = np.array([1, 1.5, 2, 3, 5, 10, 30])
per_particle["bin"] = pd.cut(per_particle.pt, bins)
eff = per_particle.groupby("bin", observed=False).found.mean()
plt.errorbar((bins[1:] + bins[:-1]) / 2, eff.values, xerr=(bins[1:] - bins[:-1]) / 2, fmt="o")
plt.xscale("log"); plt.ylim(0, 1.05); plt.xlabel("particle pT [GeV]"); plt.ylabel("double-majority efficiency"); plt.grid(alpha=0.3); plt.show()""",
        ),
        (
            "code",
            """from minihep.plotting import event_display

inputs, targets = val_ds[0]
model.eval()
with torch.no_grad():
    preds = model.predict(model({k: v.to(DEVICE) for k, v in inputs.items()}))["final"]
pv, pm = preds["track_valid"]["track_valid"][0].cpu(), preds["track_hit_valid"]["track_hit_valid"][0].cpu()
track_label = np.zeros(pm.shape[1], dtype=int)
for q in torch.nonzero(pv).flatten():
    track_label[pm[q].numpy()] = int(q) + 1
fig, axes = plt.subplots(2, 2, figsize=(12, 11))
truth_label = (targets["particle_hit_valid"][0].int().argmax(0) + 1) * targets["hit_on_valid_particle"][0]
event_display(inputs["hit_x"][0], inputs["hit_y"][0], inputs["hit_z"][0], truth_label.numpy(), "truth", axes[0])
event_display(inputs["hit_x"][0], inputs["hit_y"][0], inputs["hit_z"][0], track_label, "reconstructed", axes[1])
plt.show(); model.train()""",
        ),
        (
            "md",
            """
## Your turn

1. Set `mask_attention=False` in the decoder (edit `build_tracker` or construct by hand) and retrain. Compare
   efficiency. This is the ablation that motivated Mask2Former.
2. Increase the hit count per event (`EventConfig(mean_num_particles=60)` in `minihep/utils.py` with a new
   data folder). Time the matcher with `%timeit` on the cost matrix. How does it scale?
3. Where in hepattn does `has_first_layer_loss: false` come in, and what would happen to layer 0's loss with
   learned queries if it were true?
""",
        ),
    ]
)

# ----------------------------------------------------------------------------------------------- week 7
NOTEBOOKS["week07_lightning_pipeline.ipynb"] = nb(
    [
        (
            "md",
            """
# Week 7: the full pipeline from YAML

Normally you run these commands in a terminal (or in tmux on Hypatia). Here they run through `subprocess`
so you can see the outputs. The chain is the one you ran on TrackML: train a hit filter, write its
predictions for every split, train the tracker on the filtered hits, evaluate.
""",
        ),
        ("code", SETUP),
        (
            "code",
            """import subprocess

data_dir = ensure_toy_data(num_train=100 if FAST else 2000, num_val=20 if FAST else 200, num_test=20 if FAST else 200)
epochs = "1" if FAST else "10"

def run(*args):
    cmd = [sys.executable, "-m", *args]
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-1500:])
    if result.returncode:
        print(result.stderr[-3000:])
        raise RuntimeError("command failed")

run("minihep.run_tracking", "fit", "--config", "configs/toy_tracking.yaml", "--print_config")""",
        ),
        (
            "code",
            """run("minihep.run_filter", "fit", "--config", "configs/toy_filter.yaml", "--trainer.max_epochs", epochs, "--data.num_val", "-1")
filter_run = max(Path("logs").glob("toy-filter_*"))
for split in ("train", "val", "test"):
    run("minihep.run_filter", "test", "--config", str(filter_run / "config.yaml"), "--data.test_dir", f"data/toy/{split}")
print(sorted(p.name for p in (filter_run / "ckpts").iterdir()))""",
        ),
        (
            "code",
            """import pandas as pd

metrics = pd.read_csv(filter_run / "metrics.csv")
val = metrics.dropna(subset=["val/loss"])
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
axes[0].plot(val.epoch, val["val/loss"], "o-"); axes[0].set_title("filter val loss")
axes[1].plot(val.epoch, val["val/final_hit_filter_recall"], "o-", label="recall")
axes[1].plot(val.epoch, val["val/final_hit_filter_precision"], "o-", label="precision"); axes[1].legend()
plt.show()""",
        ),
        (
            "code",
            """import h5py

h5 = {s: next((filter_run / "ckpts").glob(f"*__{s}.h5")) for s in ("train", "val", "test")}
with h5py.File(h5["test"]) as f:
    first = sorted(f)[0]
    f[first].visititems(lambda path, obj: print(f"{first}/{path}", getattr(obj, "shape", "")))
hit_eval = [a for s in ("train", "val", "test") for a in (f"--data.hit_eval_{s}", str(h5[s]))]
run("minihep.run_tracking", "fit", "--config", "configs/toy_tracking.yaml", "--trainer.max_epochs", epochs, "--data.num_val", "-1", *hit_eval, "--data.hit_filter_threshold", "0.1")
track_run = max(Path("logs").glob("toy-tracking_*"))
run("minihep.run_tracking", "test", "--config", str(track_run / "config.yaml"))""",
        ),
        (
            "code",
            """metrics = pd.read_csv(track_run / "metrics.csv")
val = metrics.dropna(subset=["val/loss"])
fig, axes = plt.subplots(1, 3, figsize=(17, 3.5))
axes[0].plot(val.epoch, val["val/loss"], "o-"); axes[0].set_title("val/loss")
for wp in (0.5, 0.75, 1.0):
    axes[1].plot(val.epoch, val[f"val/p{wp}_eff"], "o-", label=f"eff @ {wp}")
    axes[2].plot(val.epoch, val[f"val/p{wp}_pur"], "o-", label=f"pur @ {wp}")
axes[1].legend(); axes[2].legend(); plt.show()
print((track_run / "metadata.yaml").read_text())""",
        ),
        (
            "md",
            """
## Your turn

1. Add `--config configs/comet.yaml` to the tracking fit (after putting your API key in `~/.comet.config`),
   and find the run in the Comet UI. Then rerun with `online: false` and upload the offline archive.
2. Override something deep from the command line, e.g. `--model.model.init_args.decoder.init_args.num_layers 4`
   (check the exact key with `--print_config`).
3. Write a small script that reads the tracking `__test.h5` and computes the double-majority efficiency
   per event from `preds/final/...` and `targets/...` using `minihep.metrics.double_majority`.
""",
        ),
    ]
)

# ----------------------------------------------------------------------------------------------- week 8
NOTEBOOKS["week08_precision_and_profiling.ipynb"] = nb(
    [
        (
            "md",
            """
# Week 8: precision, profiling and the NaN post-mortem
""",
        ),
        ("code", SETUP),
        (
            "md",
            """
## 1. How coarse are 16-bit floats?

The gap between neighbouring representable numbers grows with the number itself. fp16 also stops at 65504.
""",
        ),
        (
            "code",
            """x = torch.logspace(-3, 6, 400, dtype=torch.float64)
for dtype in (torch.float16, torch.bfloat16):
    nxt = torch.nextafter(x.to(dtype), torch.tensor(float("inf"), dtype=dtype)).double()
    gap = (nxt - x.to(dtype).double()) / x
    plt.loglog(x, gap, label=str(dtype))
plt.axvline(65504, color="red", ls="--", label="fp16 max")
plt.xlabel("value"); plt.ylabel("relative gap to next float"); plt.legend(); plt.grid(alpha=0.3); plt.show()""",
        ),
        (
            "md",
            """
## 2. The filter NaN, replayed

Your 30 September filter run: GradScaler starting at 4096, growth interval 2000, a corrupt event every ~1400
steps. Compare with events only every 5000 steps.
""",
        ),
        (
            "code",
            """from exercises.week08.ex01_precision import simulate_corrupt_events

for every in (5000, 1400):
    s = simulate_corrupt_events(40_000, every, growth_interval=2000, init_scale=4096.0)
    plt.semilogy(np.maximum(s.history, 1e-12), label=f"inf gradient every {every} steps")
plt.axhline(1, color="grey", ls=":")
plt.xlabel("optimiser step"); plt.ylabel("loss scale"); plt.legend(); plt.show()""",
        ),
        (
            "md",
            """
## 3. Mixed precision on your GPU

Same tracker, same data, fp32 vs bf16 autocast: time per step and peak memory.
""",
        ),
        (
            "code",
            """from exercises.week06.ex02_build_tracker import build_tracker
from minihep.data.dataset import ToyTrackingDataset

data_dir = ensure_toy_data(num_train=50)
fields = ["x", "y", "z", "r", "eta", "phi"]
ds = ToyTrackingDataset(str(data_dir / "train"), {"hit": fields}, {"hit": ["on_valid_particle"]}, event_max_num_particles=32)
inputs, targets = (({k: v.to(DEVICE) for k, v in d.items()}) for d in ds[0])
model = build_tracker(fields, dim=128, num_queries=32).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

def step(dtype):
    with torch.autocast(DEVICE, dtype=dtype, enabled=dtype != torch.float32):
        _, _, losses = model.loss(model(inputs), targets)
        loss = sum(v for layer in losses.values() for task in layer.values() for v in task.values())
    opt.zero_grad(); loss.backward(); opt.step()

from exercises.week08.ex02_debugging_and_speed import time_fn

for dtype in (torch.float32, torch.bfloat16):
    ms = time_fn(lambda: step(dtype), warmup=2, iters=5 if FAST else 20, device=DEVICE)
    print(f"{dtype!s:15s} {ms:7.1f} ms/step")""",
        ),
        (
            "md",
            """
## 4. Where does the time go? (torch.profiler)
""",
        ),
        (
            "code",
            """from torch.profiler import ProfilerActivity, profile

activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if DEVICE == "cuda" else [])
with profile(activities=activities, record_shapes=False) as prof:
    for _ in range(3):
        step(torch.float32)
sort_key = "cuda_time_total" if DEVICE == "cuda" else "cpu_time_total"
print(prof.key_averages().table(sort_by=sort_key, row_limit=15))""",
        ),
        (
            "md",
            """
Look for `linear_sum_assignment` / the matcher: on real TrackML events it dominated your training step (vault
note "Hungarian matching overhead"). On toy events it is small because the cost matrix is only 32 x ~10.

## 5. torch.compile
""",
        ),
        (
            "code",
            """import copy

eager = copy.deepcopy(model).eval()
compiled = copy.deepcopy(model).eval()
compiled.encoder.compile(dynamic=True)
with torch.no_grad():
    t_eager = time_fn(lambda: eager(inputs), warmup=2, iters=10, device=DEVICE)
    t_comp = time_fn(lambda: compiled(inputs), warmup=3, iters=10, device=DEVICE)  # the first calls compile
print(f"eager {t_eager:.2f} ms   compiled encoder {t_comp:.2f} ms")""",
        ),
        (
            "md",
            """
## Your turn

1. Make the step NaN on purpose (multiply one input by 1e6 and cast inputs to `.half()` like hepattn's data
   loader), then find the culprit with `first_nonfinite_module`.
2. Train the tracker through the CLI with `--trainer.precision bf16-mixed` and `16-mixed`. Which needs a
   GradScaler? Find it in the checkpoint: `torch.load(ckpt)["MixedPrecision"]` (this is how you diagnosed
   the scale collapse).
3. Profile a hepattn TrackML step on Hypatia with `trainer.profiler` (commented out in the TrackML configs).
""",
        ),
    ]
)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, book in NOTEBOOKS.items():
        nbf.write(book, out / name)
        print("wrote", out / name)


if __name__ == "__main__":
    main()
