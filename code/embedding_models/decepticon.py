import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable

from configurable import Configurable
from transformer.Models import Encoder


class Decepticon(nn.Module, Configurable):
    ''' An encoder with attention mechanism. '''
    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=2, n_head=2,
            d_word_vec=100, d_model=100, d_inner_hid=100, d_k=100, d_v=100,
            dropout=0.1, proj_share_weight=True):

        super(Decepticon, self).__init__()
        self.encoder = Encoder(
            n_src_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout)

        assert d_model == d_word_vec, 'To facilitate the residual connections' \
                'the dimensions of all module output shall be the same.'

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def get_sent_embedding(self, src_seq):
        src_pos = Variable(
            torch.arange(0, src_seq.size(1)).repeat(src_seq.size(0), 1).type(torch.LongTensor).cuda())
        enc_output = self.encoder(src_seq, src_pos)
        sent_embedding = enc_output.view(enc_output.size(0), -1)
        return sent_embedding

    def forward(self, src_seq):
        src_pos = Variable(
            torch.arange(0, src_seq.size(1)).repeat(src_seq.size(0), 1).type(torch.LongTensor).cuda())
        enc_output = self.encoder(src_seq, src_pos)
        sent_embedding = enc_output.view(enc_output.size(0), -1)
        return sent_embedding

    @classmethod
    def get_arguments_from_configs(cls, experiment_cfg, model_cfg):
        kwargs = {'n_max_seq': experiment_cfg.TEXT.MAX_LEN}
        kwargs.update(model_cfg)

        return kwargs


if __name__ == '__main__':
    test_module = Decepticon(10, 7)
    test_module.cuda()
    seq = np.asarray([[0, 1, 4, 5, 6, 1, 2]])
    seq = Variable(torch.from_numpy(seq).cuda())
    out = test_module(seq)
    print(out.size())
