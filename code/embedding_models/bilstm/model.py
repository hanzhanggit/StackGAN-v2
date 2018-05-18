import torch.nn as nn

from embedding_models.configurable import Configurable


class BiLSTMEncoder(nn.Module, Configurable):
    def __init__(self, n_src_vocab, d_word_vec=512, d_model=512, embedding_size=1024, dropout=0.0):
        super(BiLSTMEncoder, self).__init__()

        self.d_word_vec = d_word_vec
        self.emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=2)
        self.bilstm = nn.LSTM(d_word_vec, d_model, batch_first=True, bidirectional=True)
        self.encoding = nn.Linear(d_model * 2, embedding_size)

        # initialize weights
        self.init_weights()

    def get_trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def init_weights(self, initrange=0.1):
        self.emb.weight.data.uniform_(-initrange, initrange)
        for name, param in self.bilstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

    def forward(self, x):
        embeddings = self.emb(x)

        # take the output features (h_t) from the last layer of the BiLSTM for each t
        output, hiddens = self.bilstm(embeddings)
        encoding = self.encoding(output[:, -1, :])

        return encoding

    @classmethod
    def get_arguments_from_configs(self, exp_cfg, model_cfg):
        kwargs = {}
        kwargs.update(model_cfg)

        return kwargs


class BiLSTMClassifier(nn.Module):
    def __init__(self, n_classes,
                 n_src_vocab, d_word_vec=512, d_model=512, embedding_size=1024, dropout=0.0):
        ''' Softmax classiffier on top of BiLSTM Encoder '''
        super(BiLSTMClassifier, self).__init__()
        self.encoder = BiLSTMEncoder(n_src_vocab, d_word_vec, d_model, embedding_size, dropout)
        self.projection = nn.Linear(embedding_size, n_classes)
        self.init_weights()

    def forward(self, x):
        encoding = self.encoder(x)
        y = self.projection(encoding)
        log_probs = nn.functional.log_softmax(y, dim=1)

        return log_probs

    def init_weights(self):
        nn.init.constant(self.projection.bias, 0.0)
        nn.init.xavier_normal(self.projection.weight)

    @classmethod
    def from_configs(self, exp_cfg, model_cfg):
        return BiLSTMClassifier(**BiLSTMEncoder.get_arguments_from_configs(exp_cfg, model_cfg))
