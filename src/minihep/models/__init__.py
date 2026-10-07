"""Model building blocks. Importing them here gives YAML configs short class paths,
e.g. ``class_path: minihep.models.Encoder`` instead of ``minihep.models.encoder.Encoder``
(hepattn does the same in hepattn/models/__init__.py)."""

from minihep.models.attention import Attention
from minihep.models.decoder import MaskFormerDecoder, MaskFormerDecoderLayer
from minihep.models.dense import Dense, SwiGLU
from minihep.models.encoder import DropPath, Encoder, EncoderLayer, LayerScale, Residual
from minihep.models.hitfilter import HitFilter
from minihep.models.input import InputNet
from minihep.models.maskformer import MaskFormer
from minihep.models.matcher import Matcher
from minihep.models.posenc import PositionEncoder
from minihep.models.tasks import HitFilterTask, HitMaskTask, ObjectValidTask

__all__ = [
    "Attention",
    "Dense",
    "DropPath",
    "Encoder",
    "EncoderLayer",
    "HitFilter",
    "HitFilterTask",
    "HitMaskTask",
    "InputNet",
    "LayerScale",
    "MaskFormer",
    "MaskFormerDecoder",
    "MaskFormerDecoderLayer",
    "Matcher",
    "ObjectValidTask",
    "PositionEncoder",
    "Residual",
    "SwiGLU",
]
