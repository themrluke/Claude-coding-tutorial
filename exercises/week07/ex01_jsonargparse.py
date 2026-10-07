"""Week 7, exercise 1: jsonargparse on its own, before LightningCLI hides it.

LightningCLI is a thin layer over jsonargparse. jsonargparse turns *type hints* into a
parser: a parameter ``lr: float = 1e-3`` becomes ``--lr`` that must parse as a float; a
parameter ``model: nn.Module`` accepts ``{"class_path": ..., "init_args": {...}}`` naming
any subclass, checks the init_args against *that class's* signature, and can instantiate it.

That is why hepattn's constructors are fully type-hinted: the hints are the config schema.
"""

from pathlib import Path

import yaml
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from torch import nn


def make_parser() -> ArgumentParser:
    """Build a parser with:

    - ``--config``: a config file (``action=ActionConfigFile``); may be given several times
    - ``--name``: str, default "run"
    - ``--lr``: float, default 1e-3
    - ``--epochs``: int, default 10
    - ``--model``: any ``nn.Module`` subclass, given as class_path/init_args (``parser.add_argument("--model", type=nn.Module)``)
    - ``--tags``: list[str], default []
    """
    # >>> week07: ArgumentParser(), then six add_argument calls
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--name", type=str, default="run")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", type=nn.Module)
    parser.add_argument("--tags", type=list[str], default=[])
    return parser
    # <<< week07


def parse(args: list[str]) -> Namespace:
    """Parse ``args`` (a list like ``["--config", "a.yaml", "--lr", "0.1"]``) with make_parser. Do not instantiate."""
    # >>> week07: make_parser().parse_args(args)
    return make_parser().parse_args(args)
    # <<< week07


def build(args: list[str]) -> tuple[nn.Module, Namespace]:
    """Parse, then instantiate: return ``(model, cfg)`` where model is the constructed nn.Module.

    ``parser.instantiate_classes(cfg)`` returns a new Namespace in which every class_path block
    has been replaced by the object it describes.
    """
    # >>> week07: parse with the same parser you instantiate with
    parser = make_parser()
    cfg = parser.parse_args(args)
    init = parser.instantiate_classes(cfg)
    return init.model, cfg
    # <<< week07


def dump(args: list[str]) -> str:
    """Return the fully resolved config as YAML text (what ``--print_config`` shows): ``parser.dump(cfg)``.

    Useful to see exactly what a run will use after several configs and overrides are merged.
    Leave the ``config`` key out (``skip_none=True`` is fine; the dump drops ``config`` itself).
    """
    # >>> week07: parse, then parser.dump
    parser = make_parser()
    return parser.dump(parser.parse_args(args))
    # <<< week07


def deep_merge(base: dict, override: dict) -> dict:
    """Merge two plain config dicts the way a second ``--config`` file overrides the first.

    Nested dicts are merged key by key (recursively); anything else in ``override``, including
    lists, replaces the value in ``base``. Return a new dict; do not modify the inputs.
    """
    # >>> week07: copy base, then for each key recurse if both sides are dicts, else take override's value
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out
    # <<< week07


def load_yaml(path: Path) -> dict:
    """Read a YAML file into a dict (provided). Anchors (&x) and aliases (*x) are resolved by the YAML loader."""
    return yaml.safe_load(Path(path).read_text())
