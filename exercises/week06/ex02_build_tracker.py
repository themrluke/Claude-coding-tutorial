"""Week 6, exercise 2: assemble a complete MaskFormer tracker from your parts, by hand.

Next week the same object is built from a YAML file by LightningCLI. Writing the Python
first makes the YAML obvious: every ``class_path`` / ``init_args`` block in
src/hepattn/experiments/trackml/configs/*.yaml is one constructor call below.

Then run the overfit check (the test does it): train on 2 events until the model
reconstructs them almost perfectly. If a model cannot overfit two events, something is
wired wrong; it is the first thing to try after any architecture change. (You did exactly
this on ColliderML pu0 in September: 10 events, 3000 steps.)
"""

from torch import nn

from minihep.models.decoder import MaskFormerDecoder
from minihep.models.dense import Dense
from minihep.models.encoder import Encoder
from minihep.models.input import InputNet
from minihep.models.maskformer import MaskFormer
from minihep.models.matcher import Matcher
from minihep.models.posenc import PositionEncoder
from minihep.models.tasks import HitMaskTask, ObjectValidTask


def build_tracker(
    fields: list[str],
    dim: int = 64,
    num_queries: int = 32,
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    num_heads: int = 4,
    posenc_fields: tuple[str, ...] = ("r", "eta", "phi"),
) -> MaskFormer:
    """Build this model (names matter: the tests and next week's YAML use them):

    input_nets: ModuleList([ InputNet("hit", Dense(len(fields), dim, hidden_layers=[dim]), fields,
                                      posenc=PositionEncoder("hit", list(posenc_fields), dim, sym_fields=["phi"], alpha=10)) ])
    encoder:    Encoder(num_encoder_layers, dim, attn_kwargs={"num_heads": num_heads})
    decoder:    MaskFormerDecoder(num_queries, dim, num_decoder_layers, layer_kwargs={"attn_kwargs": {"num_heads": num_heads}})
    tasks:      ModuleList([
                  ObjectValidTask("track_valid", input_object="query", output_object="track", target_object="particle", dim=dim),
                  HitMaskTask("track_hit_valid", input_constituent="hit", input_object="query", output_object="track",
                              target_object="particle", dim=dim, losses={"mask_dice": 2.0, "mask_focal": 10.0},
                              costs={"mask_dice": 2.0, "mask_focal": 10.0}),
                ])
    matcher:    Matcher()
    and ``input_sort_field="phi"``.
    """
    # TODO(week06): construct each piece, then MaskFormer(...)
    raise NotImplementedError("week06 exercise (ex02_build_tracker.py)")
