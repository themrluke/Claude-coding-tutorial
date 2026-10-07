# Maintainer tools

* `make_stubs.py`: the `solutions` branch is the source of truth. To refresh `main` after editing
  solutions: in a checkout of `main`, `git checkout solutions -- src exercises tests configs notebooks`
  (plus any other changed files), then `python tools/make_stubs.py`, check with `pixi run lint` and
  `pixi run pytest --collect-only`, and commit.
* `build_notebooks.py`: writes `notebooks/*.ipynb` from Python source. Edit there, then
  `pixi run python tools/build_notebooks.py notebooks && pixi run ruff format notebooks && pixi run nbstripout notebooks/*.ipynb`
  (nbstripout resets the random cell ids; CI's `nbstripout --verify` fails without it). Set `MINIHEP_FAST=1` and execute them with
  `jupyter nbconvert --to notebook --execute` for a quick check that they run.
