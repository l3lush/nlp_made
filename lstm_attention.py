import torch
import torch.nn as nn
import torch.optim as optim

from torchnlp import nn as nlpnn

import random
import math
import time


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=emb_dim)

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):
        embedded = self.embedding(src)

        embedded = self.dropout(embedded)

        output, hidden = self.lstm(embedded)

        output = output[:, :, : self.hid_dim] + output[:, :, self.hid_dim :]

        return output, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
        )

        self.attn = nlpnn.Attention(hid_dim)

        self.out = nn.Linear(in_features=hid_dim * 2, out_features=output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, hidden, encoder_output):
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        output, hidden = self.lstm(embedded, hidden)

        attention_output = (
            self.attn(output.transpose(0, 1), encoder_output.transpose(0, 1))[0]
        ).transpose(0, 1)

        preds = self.out(
            torch.cat([attention_output.squeeze(0), output.squeeze(0)], dim=1)
        )

        return preds, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.linear_lt = nn.Linear(
            in_features=self.encoder.hid_dim * 2,
            out_features=self.encoder.hid_dim,
        )

        self.linear_st = nn.Linear(
            in_features=self.encoder.hid_dim * 2,
            out_features=self.encoder.hid_dim,
        )

        assert (
            encoder.hid_dim == decoder.hid_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        encoder_output, hidden = self.encoder(src)

        h1 = hidden[0].view(
            self.encoder.n_layers, src.shape[1], self.encoder.hid_dim * 2
        )
        h2 = hidden[1].view(
            self.encoder.n_layers, src.shape[1], self.encoder.hid_dim * 2
        )

        h1 = self.linear_lt(h1).contiguous()
        h2 = self.linear_st(h2).contiguous()

        hidden = (h1, h2)

        input = trg[0, :]

        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden, encoder_output)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = trg[t] if teacher_force else top1

        return outputs
