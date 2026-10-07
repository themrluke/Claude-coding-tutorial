"""Week 1, exercise 3: abstract base classes and the dict-key convention.

Two ideas that you need before hepattn's model code makes sense.

1. **Everything travels in dicts keyed by strings**, and the keys follow a pattern:

       "{input_name}_{field}"     hit_x, hit_valid, hit_embed, hit_on_valid_particle
       "{object}_{field}"         particle_valid, particle_pt, track_logit
       "{object}_{input}_{field}" particle_hit_valid, track_hit_logit

   That is why MaskFormer asserts ``not any("_" in name for name in self.input_names)``:
   an underscore in an input name would make the keys ambiguous. See
   src/hepattn/models/maskformer.py (forward) and src/hepattn/utils/model_utils.py.

2. **Tasks are subclasses of an abstract base class** (src/hepattn/models/task.py).
   ``Task`` declares ``forward``, ``predict`` and ``loss`` as ``@abstractmethod``, so
   Python refuses to create a task that forgot one of them, and gives harmless
   defaults for the optional hooks (``cost``, ``attn_mask``, ``metrics``...).

We use plain Python lists here; the tensor versions come in week 2.
"""

from abc import ABC, abstractmethod


def split_key(key: str, input_names: list[str]) -> tuple[str, str]:
    """Split ``"hit_on_valid_particle"`` into ``("hit", "on_valid_particle")``.

    Only names in ``input_names`` count as prefixes. If no name matches, raise
    ``KeyError``. Note "hit" must not match "hitx_y": the prefix has to be followed
    by an underscore.
    """
    # TODO(week01): find the input name that, followed by "_", starts the key
    raise NotImplementedError("week01 exercise (ex03_abc_and_naming.py)")


def merge_inputs(x: dict[str, list], input_names: list[str]) -> dict[str, list]:
    """Concatenate per-input embeddings into one "key" sequence, the way MaskFormer.forward does.

    Input: ``x`` holds ``f"{name}_embed"`` and ``f"{name}_valid"`` lists for every name.
    Return a *new* dict containing everything in ``x`` plus:

      - ``"key_embed"``: all ``{name}_embed`` lists joined, in ``input_names`` order
      - ``"key_valid"``: all ``{name}_valid`` lists joined, same order
      - ``f"key_is_{name}"`` for every name: a list of bools, same length as
        ``key_embed``, True where that element came from ``name``

    Example with input_names ["pix", "strip"], pix_embed [1, 2], strip_embed [3]:
      key_embed [1, 2, 3], key_is_pix [True, True, False], key_is_strip [False, False, True]
    """
    # TODO(week01): build the joined lists and one membership mask per input name
    raise NotImplementedError("week01 exercise (ex03_abc_and_naming.py)")


def unmerge_inputs(x: dict[str, list], input_names: list[str]) -> dict[str, list]:
    """Inverse of ``merge_inputs``: copy ``key_embed`` back into each ``{name}_embed``.

    Use the ``key_is_{name}`` masks to pick the right elements. Return a new dict.
    (Real version: hepattn.utils.model_utils.unmerge_inputs, which runs after the
    encoder and after every decoder layer.)
    """
    # TODO(week01): for each name, keep the key_embed elements whose key_is_{name} entry is True
    raise NotImplementedError("week01 exercise (ex03_abc_and_naming.py)")


class Task(ABC):
    """A cut-down version of hepattn's Task base class. Provided, nothing to do here."""

    def __init__(self, name: str, input_object: str):
        self.name = name
        self.input_object = input_object

    @abstractmethod
    def forward(self, x: dict[str, list]) -> dict[str, list]:
        """Turn embeddings into raw outputs (logits)."""

    @abstractmethod
    def predict(self, outputs: dict[str, list]) -> dict[str, list]:
        """Turn raw outputs into predictions (e.g. booleans after a threshold)."""

    def attn_mask(self, outputs: dict[str, list]) -> dict[str, list]:
        """Optional hook: most tasks do not produce an attention mask."""
        return {}


class ThresholdTask(Task):
    """A task that scores each element of ``{input_object}_embed`` and thresholds it.

    - ``forward(x)`` returns ``{f"{input_object}_score": [w * e for e in x[f"{input_object}_embed"]]}``
    - ``predict(outputs)`` returns ``{f"{input_object}_pred": [s >= threshold for s in scores]}``

    Call ``super().__init__`` so ``name`` and ``input_object`` are set by the base class.
    """

    def __init__(self, name: str, input_object: str, weight: float = 1.0, threshold: float = 0.5):
        # TODO(week01): call the base class constructor, then store weight and threshold
        raise NotImplementedError("week01 exercise (ex03_abc_and_naming.py)")

    def forward(self, x: dict[str, list]) -> dict[str, list]:
        # TODO(week01): score every element of the input embedding
        raise NotImplementedError("week01 exercise (ex03_abc_and_naming.py)")

    def predict(self, outputs: dict[str, list]) -> dict[str, list]:
        # TODO(week01): threshold the scores
        raise NotImplementedError("week01 exercise (ex03_abc_and_naming.py)")
