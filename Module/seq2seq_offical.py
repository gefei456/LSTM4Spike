#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:         14:24
# @Author:      WGF
# @File:        seq2seq_offical.py
# @Description:


from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU_Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_encoder_layers, n_decoder_layers, device, dropout_p=0.1):
        super(GRU_Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        # Initiate encoder and Decoder layers
        # self.encoder = nn.ModuleList()
        # for i in range(n_encoder_layers):
        self.encoder = EncoderRNN(input_size, hidden_size, n_encoder_layers, device, dropout_p)

        # self.decoder = nn.ModuleList()
        # for i in range(n_decoder_layers):
        self.decoder = DecoderRNN(hidden_size, output_size, n_decoder_layers, device, dropout_p)

    def forward(self, X, Y=None):
        enc_output, enc_hidden = self.encoder(X)
        dec_output, dec_hidden = self.decoder(enc_output, enc_hidden, Y)
        return dec_output


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, device, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.device = device

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers, device, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        # self.embedding = nn.Embedding(output_size, hidden_size)
        self.output_size = output_size
        self.embedding = nn.Linear(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.device = device

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        decoder_input = torch.empty(batch_size, 1, self.output_size, dtype=torch.float32, device=self.device).fill_(1)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(seq_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)    # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                # _, topi = decoder_output.topk(1)
                decoder_input = decoder_output.detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # decoder_outputs = F.tanh(decoder_outputs)
        return decoder_outputs, decoder_hidden # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        # 新模型采用此行代码
        output = self.dropout(self.embedding(input))
        # 旧模型没有dropout
        # output = self.embedding(input)
        # output = F.tanh(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
