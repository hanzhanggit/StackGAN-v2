import torch.nn as nn

from configurable import Configurable


class EmbeddingBlock(nn.Module, Configurable):
    def __init__(self, n_src_vocab, embedding_dim, initrange=0.1):
        super(EmbeddingBlock, self).__init__()
        self.emb = nn.Embedding(n_src_vocab, embedding_dim, padding_idx=2)
        self.emb.weight.data.uniform_(-initrange, initrange)

    def forward(self, sequence):
        embeddings = self.emb(sequence)
        # return embeddings.mean(1) #mean
        print(embeddings.view(embeddings.size(0), -1).size())
        return embeddings.view(embeddings.size(0), -1)

    @classmethod
    def get_arguments_from_configs(self, experiment_cfg, model_cfg):
        kwargs = {'embedding_dim': experiment_cfg.TEXT.DIMENSION / experiment_cfg.TEXT.MAX_LEN}
        kwargs.update(model_cfg)

        return kwargs
