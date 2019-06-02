# -*- coding: utf-8 -*-

"""
    Implementation of the following:
        1. Pointer Network: ...
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Attention


class PointerNetwork(nn.Module):
    """
    Add class docstring.
    """
    def __init__(self, args):
        super(PointerNetwork, self).__init__()

        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.seq_len = args.seq_len
        self.n_glimpses = args.n_glimpses
        self.tanh_exploration = args.tanh_exploration
        self.use_tanh = args.use_tanh
        self.use_gpu = args.use_gpu and torch.cuda.is_available()

        self.rnn_type = args.rnn_type.lower()
        if self.rnn_type == 'lstm':
            rnn = nn.LSTM
        elif self.rnn_type == 'gru':
            rnn = nn.GRU

        self.embedding = nn.Embedding(self.seq_len, self.embedding_size)
        self.encoder = rnn(self.embedding_size, self.hidden_size, batch_first=True)
        self.decoder = rnn(self.embedding_size, self.hidden_size, batch_first=True)
        self.pointer = Attention(self.hidden_size, use_tanh=self.use_tanh, c=self.tanh_exploration)
        self.glimpse = Attention(self.hidden_size, use_tanh=False)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(self.embedding_size))
        self.decoder_start_input.data.uniform_(
            -(1. / np.sqrt(self.embedding_size)), 1. / np.sqrt(self.embedding_size)
        )

        self.criterion = nn.CrossEntropyLoss()

    def apply_mask_to_logits(self, logits, mask, idxs, inf=10):
        """
        Mask out idxs for further softmax calculation.
        Arguments:
            logits: tensor of shape (batch_size, seq_len)
            mask: tensor of shape (batch_size, seq_len)
            idx: tensor of shape (batch_size, )
        """

        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask = clone_mask.scatter_(
                dim=1,
                index=idxs.view(-1, 1),
                value=1
            )
            clone_mask = clone_mask.byte()
            logits[clone_mask] = -inf

        return logits, clone_mask

    def forward(self, inputs, target):
        """
        Arguments:
            inputs: tensor of shape (batch_size, seq_len)
            target: ...
        """

        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        assert seq_len == self.seq_len

        embedded = self.embedding(inputs)
        target_embedded = self.embedding(target)

        encoder_outputs, hidden = self.encoder(embedded)

        mask = torch.zeros(batch_size, seq_len).byte()
        if self.use_gpu:
            mask = mask.cuda()

        decoder_input = self.decoder_start_input.unsqueeze(0)  # (1, embedding_size)
        decoder_input = decoder_input.repeat(batch_size, 1)    # (batch_size, embedding_size)

        idxs = None
        loss = 0.

        for i in range(seq_len):

            decoder_input = decoder_input.unsqueeze(1)         # (batch_size, 1, embedding_size)
            _, hidden = self.decoder(decoder_input, hidden)    # _, (1, batch_size, hidden_size)

            if self.rnn_type == 'lstm':
                query = hidden[0]     # LSTM outputs: outputs, (h_t, c_t)
            elif self.rnn_type == 'gru':
                query = hidden        # GRU outputs: outputs, h_t
            else:
                raise NotImplementedError

            query = query.squeeze(0)  # (1, batch_size, hidden_size) -> (batch_size, hidden_size)

            for _ in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, None)

                softmax_values = F.softmax(logits, dim=1)
                softmax_values = softmax_values.unsqueeze(2)  # (batch_size, seq_len, 1)

                query = torch.bmm(ref, softmax_values)        # (batch_size, hidden_size, 1)
                query = query.squeeze(2)                      # (batch_size, hidden_size)

            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)

            loss += self.criterion(logits, target[:, i])

            idxs = logits.argmax(dim=1)
            decoder_input = target_embedded[:, i, :]

        return loss / seq_len
