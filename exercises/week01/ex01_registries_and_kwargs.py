"""Week 1, exercise 1: registries, **kwargs forwarding and mutable defaults.

hepattn builds almost everything from strings in a YAML file. The trick that
makes this possible is a *registry*: a plain dict from a name to a class or a
function. Look at these in the real repo before starting:

    src/hepattn/models/norm.py        NORM_TYPES = {"LayerNorm": nn.LayerNorm, ...}
    src/hepattn/models/attention.py   ATTN_TYPES = {"torch": scaled_dot_product_attention, ...}
    src/hepattn/models/loss.py        cost_fns / loss_fns
    src/hepattn/models/matcher.py     SOLVERS

The second trick is forwarding keyword arguments through several layers of
constructors (`Encoder(**layer_kwargs)` -> `EncoderLayer(attn_kwargs=...)` ->
`Attention(**attn_kwargs)`). See `Encoder.__init__` in models/encoder.py.

Run the tests with:  pixi run week 01
"""

from dataclasses import dataclass, field


# A few toy "layers" so the registry has something to build.
@dataclass
class LayerNorm:
    dim: int
    eps: float = 1e-5


@dataclass
class RMSNorm:
    dim: int
    eps: float = 1e-6


@dataclass
class DyT:
    dim: int
    alpha_init_value: float = 0.5


NORM_TYPES: dict[str, type] = {
    "LayerNorm": LayerNorm,
    "RMSNorm": RMSNorm,
    "DyT": DyT,
}


def build_norm(name: str | None, dim: int, **kwargs):
    """Build a norm layer from its registry name.

    - ``name=None`` means "no norm": return None.
    - An unknown name must raise ``ValueError`` whose message contains every valid
      name, so a typo in a YAML file tells you what you could have written.
      (hepattn: ``f"Unsupported norm: {norm}. Must be one of {list(NORM_TYPES.keys())}"``)
    - Extra keyword arguments are forwarded to the class.

    >>> build_norm("RMSNorm", 8, eps=1e-3)
    RMSNorm(dim=8, eps=0.001)
    """
    # >>> week01: return None for None, raise ValueError for unknown names, else look up and call the class
    if name is None:
        return None
    if name not in NORM_TYPES:
        raise ValueError(f"Unsupported norm: {name}. Must be one of {list(NORM_TYPES.keys())}")
    return NORM_TYPES[name](dim, **kwargs)
    # <<< week01


@dataclass
class AttentionConfig:
    dim: int
    attn_type: str = "torch"
    window_size: int | None = None
    num_heads: int = 8
    bias: bool = True
    is_first_layer: bool = False


@dataclass
class LayerConfig:
    depth: int
    norm: str
    attn: AttentionConfig
    dense_kwargs: dict = field(default_factory=dict)


FLASH_ATTN_TYPES = ("flash", "flash-varlen")


def build_layer_configs(
    num_layers: int,
    dim: int,
    attn_type: str = "torch",
    window_size: int | None = None,
    **layer_kwargs,
) -> list[LayerConfig]:
    """Mimic how ``hepattn.models.encoder.Encoder.__init__`` hands options down to its layers.

    ``layer_kwargs`` may contain:
      - ``norm`` (str, default "LayerNorm")
      - ``attn_kwargs`` (dict or None): extra options for AttentionConfig, e.g. {"num_heads": 4}
      - ``dense_kwargs`` (dict or None)

    Rules (the same ones the real Encoder follows):
      1. ``attn_type`` always ends up in every layer's attention config.
      2. ``window_size`` is only passed to attention when ``attn_type`` is one of
         FLASH_ATTN_TYPES; otherwise the attention gets ``window_size=None``
         (flex expresses its window through a mask instead).
      3. ``is_first_layer`` is True only for depth 0.
      4. Every layer must get its *own* dicts. Mutating layer 0's ``dense_kwargs``
         must not change layer 1's. (The real code gets away with sharing because it
         reads the dict immediately; here you return the configs, so sharing would bite.)
      5. Passing ``attn_kwargs=None`` must behave like passing ``{}``.
    """
    # >>> week01: copy attn_kwargs/dense_kwargs per layer; set attn_type, window_size (flash only) and is_first_layer
    norm = layer_kwargs.get("norm", "LayerNorm")
    base_attn_kwargs = layer_kwargs.get("attn_kwargs") or {}
    base_dense_kwargs = layer_kwargs.get("dense_kwargs") or {}
    configs = []
    for depth in range(num_layers):
        attn_kwargs = dict(base_attn_kwargs)
        attn_kwargs["attn_type"] = attn_type
        attn_kwargs["window_size"] = window_size if attn_type in FLASH_ATTN_TYPES else None
        attn_kwargs["is_first_layer"] = depth == 0
        configs.append(LayerConfig(depth=depth, norm=norm, attn=AttentionConfig(dim=dim, **attn_kwargs), dense_kwargs=dict(base_dense_kwargs)))
    return configs
    # <<< week01


def collect_names(name: str, names: list[str] | None = None) -> list[str]:
    """Append ``name`` to ``names`` and return the list.

    If ``names`` is not given, start a fresh list *every call*. The obvious way to
    write this, ``def collect_names(name, names=[])``, is a classic Python bug: the
    default list is created once and shared by every call. hepattn always writes
    ``attn_kwargs: dict | None = None`` followed by ``attn_kwargs = attn_kwargs or {}``
    for exactly this reason.
    """
    # >>> week01: create a new list when names is None
    if names is None:
        names = []
    names.append(name)
    return names
    # <<< week01
