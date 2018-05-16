from bilstm.model import BiLSTMEncoder
from pretrained import EmbeddingBlock
from decepticon import Decepticon

from registry import register


__all__ = [
    register(BiLSTMEncoder, name="bilstm"),
    register(Decepticon, name="decepticon"),
    register(EmbeddingBlock, name="pretrained", default=True)
]
