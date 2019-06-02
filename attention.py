# -*- coding: utf-8 -*-

"""
    Implementation of the following:
        1.
        2.
        3.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Add class docstring."""
    def __init__(self, hidden_size, use_tanh=False, c=10):
        super(Attention, self).__init__()

        self.use_tanh = use_tanh
        self.query_op = nn.Linear(hidden_size, hidden_size)      # Linear transform of decoder output
        self.ref_op = nn.Conv1d(hidden_size, hidden_size, 1, 1)  # 1d convolution over encoder outputs
        self.c = c

        self.use_gpu = torch.cuda.is_available()

        v = torch.FloatTensor(hidden_size)

        # Define as trainable weights
        self.v = nn.Parameter(v, requires_grad=True)
        self.v.data.uniform_(
            -(1. / np.sqrt(hidden_size)), 1. / np.sqrt(hidden_size)
        )

    def forward(self, query, ref):
        """
        Arguments:
            query: decoder output of shape (batch_size, hidden_size)
            ref: encoder outputs of shape (batch_size, seq_len, hidden_size)
        Returns:
            ref: transformed encoder outputs of shape (batch_size, hidden_size, seq_len)
            logits: logit scores of shape (batch_size, seq_len)
        """

        batch_size = ref.size(0)
        seq_len = ref.size(1)

        ref = ref.permute(0, 2, 1)    # (batch_size, hidden_size, seq_len)
        ref = self.ref_op(ref)        # 1d conv, considering feature dimension as channels

        query = self.query_op(query)  # q_out = W_q * q_in (q_in = decoder output)
        query = query.unsqueeze(2)    # (batch_size, hidden_size, 1)

        expanded_query = query.repeat(1, 1, seq_len)                  # (batch_size, hidden_size, seq_len)
        expanded_v = self.v.view(1, 1, -1).repeat(batch_size, 1, 1)   # (batch_size, 1, hidden_size)

        logits = torch.bmm(expanded_v, torch.tanh(expanded_query + ref))  # (batch_size, 1, seq_len)
        logits = logits.squeeze(1)                                        # (batch_size, seq_len)

        if self.use_tanh:
            logits = self.c * F.tanh(logits)

        return ref, logits
