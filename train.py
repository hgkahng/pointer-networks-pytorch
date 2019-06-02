# -*- coding: utf-8 -*-

"""
    Training Pointer Network for:
        1. Sorting
        2. Convex Hull problem
        3. Traveling Salesman Problem
        4. ...
    https://github.com/higgsfield/np-hard-deep-reinforcement-learning
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from datasets import SortDataset
from models import PointerNetwork


def parse_args():
    """Parsing command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='sort', help='')

    parser.add_argument('--embedding_size', type=int, default=64, help='')
    parser.add_argument('--hidden_size', type=int, default=32, help='')
    parser.add_argument('--seq_len', type=int, default=10, help='')
    parser.add_argument('--n_glimpses', type=int, default=1, help='')
    parser.add_argument('--use_tanh', type=bool, default=False, help='')
    parser.add_argument('--tanh_exploration', type=float, default=10., help='')
    parser.add_argument('--rnn_type', type=str, default='lstm', help='lstm or gru')
    parser.add_argument('--use_gpu', type=bool, default=True, help='')

    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--epochs', type=int, default=30, help='')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    if args.dataset == 'sort':
        train_loader = DataLoader(
            dataset=SortDataset(
                num_samples=1000,
                length=args.seq_len,
            ),
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            dataset=SortDataset(
                num_samples=100,
                length=args.seq_len,
            ),
            batch_size=1,
            shuffle=False,
        )
    else:
        raise NotImplementedError

    steps_per_epoch = len(train_loader.dataset) // args.batch_size

    model = PointerNetwork(args)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if args.use_gpu:
        model.cuda()

    # Train
    model.train()
    train_loss = []

    for epoch in range(args.epochs):
        for i, batch in enumerate(train_loader):

            inputs = batch
            target = batch.sort()[1]
            if args.use_gpu:
                inputs = inputs.cuda()
                target = target.cuda()

            loss = model(inputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            if i % 10 == 0:
                print("Epoch [{:>4}/{:>4}] | ".format(epoch, args.epochs), end='')
                print("Step: [{:>4}/{:>4}] | ".format(i, steps_per_epoch), end='')
                print("Train loss: {:.6f}".format(loss.item()), end='\n')
            else:
                continue


if __name__ == '__main__':
    main()
    print('Finished training...')
