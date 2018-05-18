import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable


class fixMaskEmbeddedDropout(nn.Module):
    def __init__(self, embed, dropout=0.5):
        super(fixMaskEmbeddedDropout, self).__init__()
        self.dropout = dropout
        self.e = embed
        w = getattr(self.e, 'weight')
        del self.e._parameters['weight']
        self.e.register_parameter('weight_raw', Parameter(w.data))

    def _setweights(self):
        raw_w = getattr(self.e, 'weight_raw')
        if self.training:
            mask = raw_w.data.new().resize_((raw_w.size(0), 1)).bernoulli_(1 - self.dropout).expand_as(raw_w) / (1 - self.dropout)
            w = Variable(mask) * raw_w
            setattr(self.e, 'weight', w)
        else:
            setattr(self.e, 'weight', Variable(raw_w.data))

    def forward(self, draw_mask, *args):
        if draw_mask or not self.training:
            self._setweights()
        return self.e.forward(*args)


if __name__ == '__main__':
    V = 50
    h = 4
    bptt = 10
    batch_size = 2

    e = nn.Embedding(V, h)
    f = nn.Embedding(V, h)
    f.weight.data = e.weight.data.clone()
    embed_drop = fixMaskEmbeddedDropout(f)

    words = np.random.random_integers(low=0, high=V - 1, size=(batch_size, bptt))
    words = torch.LongTensor(words)
    words = Variable(words)

    print("0")
    print(e(words))
    embed_drop.eval()
    print("1 - should be the same as 0")
    print(embed_drop(True, words))
    print("2 - should be the same as 1")
    print(embed_drop(False, words))
    embed_drop.train()
    print("3 - should be different than 2")
    print(embed_drop(True, words))
    print("4 - should be different than 3")
    print(embed_drop(True, words))
    print("5 - should be the same as 4")
    print(embed_drop(False, words))
    embed_drop.eval()
    print("6 - should be the same as 0")
    print(embed_drop(False, words))
    print("7 - should be the same as 0")
    print(embed_drop(True, words))
