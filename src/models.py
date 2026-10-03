"""Recurrent text classifiers built from scratch in PyTorch.

All variants share one architecture so the comparison is fair:

    token ids -> Embedding -> RNN cell (RNN | LSTM | GRU, optionally
    bidirectional) -> pooled sentence vector -> Dropout -> Linear -> logits

Pooling concatenates the final hidden state with a masked max-pool over all
time steps. The max-pool lets the model react to a single strong word
("downgrade", "beats") anywhere in the tweet, which helps the plain RNN in
particular because its final state forgets early tokens.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import config

RNN_TYPES = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


class RNNClassifier(nn.Module):
    """Embedding + recurrent encoder + linear head for 3-class sentiment."""

    def __init__(self, vocab_size, rnn_type="lstm", embed_dim=config.EMBED_DIM,
                 hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS,
                 bidirectional=False, dropout=config.DROPOUT,
                 num_classes=config.NUM_CLASSES, pad_idx=0):
        super().__init__()
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout / 2)
        self.rnn = RNN_TYPES[rnn_type](
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0)
        directions = 2 if bidirectional else 1
        # final hidden state + max-pooled outputs
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim * directions, num_classes)

    def forward(self, ids, lengths):
        emb = self.emb_dropout(self.embedding(ids))
        # Packing lets the RNN skip padding so the final state is the true
        # last token rather than a pad position.
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, hidden = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        if self.rnn_type == "lstm":
            hidden = hidden[0]  # (h_n, c_n) -> h_n
        if self.rnn.bidirectional:
            last = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last = hidden[-1]
        # Masked max-pool over valid time steps
        mask = (torch.arange(out.size(1), device=ids.device)[None, :]
                < lengths.to(ids.device)[:, None])
        out = out.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(torch.cat([last, pooled], dim=1)))


# Named variants trained by ``run_baselines`` (kwargs -> RNNClassifier)
BASELINE_VARIANTS = {
    "SimpleRNN": dict(rnn_type="rnn"),
    "LSTM": dict(rnn_type="lstm"),
    "GRU": dict(rnn_type="gru"),
    "BiLSTM": dict(rnn_type="lstm", bidirectional=True),
}
