# hepattn-course

An 8-week, ~4 hours/week course that takes you from "I can write a Python class" to reading,
modifying and writing code like [hepattn](https://github.com/samvanstroud/hepattn).

You build **minihep**, a small copy of hepattn, piece by piece: the data pipeline, attention
from scratch, a transformer encoder, flash/flex/windowed attention, a MaskFormer decoder,
Hungarian matching, a Lightning + LightningCLI training setup, and finally mixed precision,
`torch.compile` and profiling. Every minihep module mirrors a hepattn module with the same
names, so after each week you read the real file and recognise it.

The **lessons** (reading material) live in the Obsidian vault under
`Learning/Coding Tutorial/`. This repository holds the **code**: exercises, tests,
notebooks and the minihep package.

## Course map

| Week | Topic | Exercises (fill in the `TODO(weekNN)` blocks) | Notebook |
|---|---|---|---|
| 1 | Tooling (WSL, pixi, git, pytest, ruff) and the Python hepattn is written in | `exercises/week01/*` | `week01_reading_hepattn` |
| 2 | NumPy/PyTorch tensor moves; parquet, HDF5 and the Dataset | `exercises/week02/*`, `src/minihep/data/{prep,dataset}.py` | `week02_toy_detector` |
| 3 | `nn.Module`, autograd, losses, optimisers, a training loop by hand | `exercises/week03/*`, `models/{dense,posenc,input,loss,tasks,hitfilter}.py` | `week03_mlp_hit_filter` |
| 4 | Attention and the transformer encoder from scratch | `exercises/week04/*`, `models/{norm,attention,encoder}.py` | `week04_transformer_filter` |
| 5 | Efficient attention: online softmax, FlashAttention, varlen, FlexAttention, windows, sorting | `exercises/week05/*`, `flex/masks.py`, flex/flash parts of `attention.py`, `encoder.py` | `week05_efficient_attention` |
| 6 | MaskFormer, Hungarian matching, mask losses, tracking metrics | `exercises/week06/*`, `models/{loss,matcher,tasks,decoder,maskformer}.py`, `metrics.py` | `week06_maskformer` |
| 7 | Lightning, LightningCLI/jsonargparse configs, callbacks, HDF5 predictions, Comet | `exercises/week07/*`, `lightning/*`, `data/datamodule.py`, `cli.py` | `week07_lightning_pipeline` |
| 8 | Mixed precision, NaN hunting, profiling, `torch.compile`, advanced git, capstone on hepattn | `exercises/week08/*`, `Compile` in `lightning/callbacks.py` | `week08_precision_and_profiling` |

Each week: read the vault lesson (~1 h), do the exercises until `pixi run week NN` is green
(~2.5 h), run the notebook and do its "Your turn" cells (~0.5 h).

## Setup (laptop, WSL2, NVIDIA GPU)

1. **WSL2 + GPU.** Install the normal NVIDIA driver *on Windows* (not inside Linux). In an Ubuntu
   WSL terminal, `nvidia-smi` should list your GPU.
2. **pixi** (installs Python, torch and everything else; no conda or pip needed):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | sh     # then open a new terminal
   ```
3. **Clone and install** (keep the repo inside the Linux filesystem, e.g. `~/code`, not `/mnt/c`; it is much faster):
   ```bash
   git clone git@github.com:themrluke/Claude-coding-tutorial.git ~/code/hepattn-course
   cd ~/code/hepattn-course
   pixi install --locked          # exactly the versions in pixi.lock
   pixi run gpu-check             # torch version, True, your GPU's name
   pixi run week 01               # week 1's tests: they fail until you write the code
   ```
   Optional environments: `pixi install -e fa2` adds FlashAttention-2 (needs an Ampere or newer GPU:
   RTX 30xx/40xx/50xx); `-e cpu` is a CPU-only torch.
4. **Editor.** VS Code with the *WSL* and *Python* extensions; open the folder from WSL
   (`code .`) and pick the interpreter `.pixi/envs/default/bin/python`.
5. **Notebooks:** `pixi run lab` and open the URL it prints, or open the `.ipynb` in VS Code with the same interpreter.
6. **Pre-commit hooks** (format and lint on every commit, like hepattn): `pixi run pre-commit install`.

### GPUs

Everything runs on any CUDA GPU from the last few generations; the course was planned for an
**RTX 4070 Ti** (Ada, 12 GB) and an **RTX 3050 Ti Laptop** (Ampere, 4 GB). Both support bf16,
`torch.compile`/Triton, FlexAttention and FlashAttention-2 (the `fa2` environment).

* Needs a recent Windows NVIDIA driver (CUDA 12.8 support, i.e. driver 570 or newer). Check the
  "CUDA Version" in the top right of `nvidia-smi` inside WSL.
* Where the GPU is used: the notebooks pick `cuda` automatically for training (weeks 3, 4, 6, 8),
  the week 5 kernel benchmark and week 8's mixed-precision and profiling sections; the week 7 CLI
  runs use `accelerator: auto`. Tests marked `gpu`/`flash` run on the GPU when there is one and
  skip otherwise; the rest of the tests are deliberately small CPU tests so they run in seconds.
* On the 4 GB laptop GPU everything in the course fits. If you push the week 5 benchmark beyond
  its default sizes or enlarge the models, close other GPU programs first; on an out-of-memory
  error halve the size.
* The 4070 Ti is the better machine for the benchmark and training notebooks; the laptop is fine
  for exercises and tests.

## How the exercises work

* Code you write sits where you see
  ```python
  # TODO(week04): unsqueeze, get_omegas(...), sin/cos, cat
  raise NotImplementedError("week04 exercise (posenc.py)")
  ```
  Delete the `raise`, write the code. Docstrings say exactly what is wanted and point at the
  hepattn file it mirrors.
* Tests: `pixi run week 04` runs one week; `pixi run test` runs everything;
  `pixi run pytest tests/week04 -k rmsnorm -x` runs one test and stops at the first failure.
  Slow tests are marked: `-m "not slow"` skips them. GPU tests skip themselves without a GPU.
* Later weeks build on earlier ones (week 6's MaskFormer uses your week 4 Attention). If you are
  stuck on an earlier piece, take that one file from the solutions (below) and keep going.

## Solutions (only when you want them)

The finished code lives on the **`solutions`** branch, never on `main`. On that branch the part you
had to write is wrapped in `# >>> weekNN` / `# <<< weekNN` markers.

```bash
git fetch origin
git diff main origin/solutions -- src/minihep/models/attention.py       # just the answer for one file
git show origin/solutions:exercises/week05/ex01_online_softmax.py        # print a whole solution file
git restore --source origin/solutions -- src/minihep/models/norm.py      # take it into your working tree
git worktree add ../hepattn-course-solutions solutions                   # a second checkout, side by side
```

## Data

The toy detector (`src/minihep/data/toy_detector.py`) simulates charged particles curling in a
2 T field through 8 barrel layers, with noise, smearing and inefficiency: TrackML in miniature,
small enough that everything trains on a laptop in minutes.

```bash
pixi run python -m minihep.data.prep --out data/toy --train 2000 --val 200 --test 200 --hdf5
```

(The notebooks call `minihep.utils.ensure_toy_data()`, which does the same.) `data/` and `logs/` are git-ignored.

## minihep ↔ hepattn

| minihep | hepattn |
|---|---|
| `data/dataset.py`, `data/datamodule.py` | `experiments/trackml/data.py` |
| `models/attention.py` | `models/attention.py` |
| `models/encoder.py`, `models/norm.py`, `models/dense.py`, `models/posenc.py`, `models/input.py` | same names in `models/` |
| `flex/masks.py` | `flex/sliding_window.py`, `flex/local_ca.py` |
| `models/tasks.py` | `models/task.py` |
| `models/loss.py`, `models/matcher.py` | same names in `models/` |
| `models/decoder.py`, `models/maskformer.py`, `models/hitfilter.py` | same names in `models/` |
| `lightning/wrapper.py` | `models/wrapper.py`, `experiments/trackml/run_tracking.py` |
| `lightning/callbacks.py` | `callbacks/` |
| `cli.py`, `run_tracking.py`, `run_filter.py` | `utils/cli.py`, `experiments/trackml/run_*.py` |
| `configs/toy_tracking.yaml` | `experiments/trackml/configs/tracking-*.yaml` |

## Repository layout

```
configs/        YAML for the LightningCLI (week 7)
exercises/      stand-alone drills, one folder per week
notebooks/      one notebook per week
src/minihep/    the package you build
tests/          one folder per week; tests/conftest.py has shared fixtures
tools/          make_stubs.py turns the solutions into exercises; build_notebooks.py writes notebooks/
```
