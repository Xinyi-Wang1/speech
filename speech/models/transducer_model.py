from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import copy
import json

import transducer.decoders as td
import transducer.functions.transducer as transducer
from . import model

class Transducer(model.Model):
    def __init__(self, freq_dim, vocab_size, config):
        super().__init__(freq_dim, config)

        # For decoding
        decoder_cfg = config["decoder"]
        rnn_dim = self.encoder_dim
        embed_dim = decoder_cfg["embedding_dim"]
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dec_rnn = nn.GRU(input_size=embed_dim,
                              hidden_size=rnn_dim,
                              num_layers=decoder_cfg["layers"],
                              batch_first=True, dropout=config["dropout"])

        global bias_data
        bias_data, num_bias_input = self.prepBiasData() # need to give z an extra dimension(batch) than x?
        size_bias = bias_data.size()
        bias_cnn_dim = self.calculate_bias_cnn(size_bias)
        self.biasEncode = biasEncoder(input_size = num_bias_input, output_size = rnn_dim, embedding_dim = embed_dim, bias_cnn_dim=bias_cnn_dim)

        # include the blank token
        self.blank = vocab_size
        self.fc1 = model.LinearND(rnn_dim, rnn_dim)
        self.fc2 = model.LinearND(rnn_dim, vocab_size + 1)

    def calculate_bias_cnn(self, size_bias, batch_size=8):
        numElement = np.prod(size_bias)
        dim = int(numElement / batch_size)
        return dim

    def prepBiasData(self):
        trainfilename = "/Users/xinyiwang/Documents/GitHub/speech/examples/timit/data/contextTrain.json"
        testfilename = "/Users/xinyiwang/Documents/GitHub/speech/examples/timit/data/contextTest.json"
        listOfTensorsTrain, numInputTrain = self.preprocess(trainfilename)
        listOfTensorsTest, numInputTest = self.preprocess(testfilename)
        listOfTensors = listOfTensorsTrain + listOfTensorsTest
        num_bias_input = numInputTrain + numInputTest
        inputs = self.zero_pad_concat_bias(listOfTensors)
        tensor1 = torch.from_numpy(inputs)
        dim8 = tensor1.view(8, 1, -1)
        return dim8, num_bias_input

    def forward(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        y_mat = self.label_collate(batch[1])
        return self.forward_impl(x, y_mat)

    def forward_impl(self, x, y):
        if self.is_cuda:
            x = x.cuda()
            y = y.cuda()
        x = self.encode(x)
        shape = x.size()
        z, _ = self.collate_bias(bias_data)
        z = self.biasEncode(z)
        out = self.decode(x, y, z)
        return out

    def loss(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        y_mat = self.label_collate(batch[1])
        out = self.forward_impl(x, y_mat)
        loss_fn = transducer.TransducerLoss()
        loss = loss_fn(out, y, x_lens, y_lens)
        return loss

    def decode(self, x, y, z):
        """
        x should be shape (batch, time, hidden dimension)
        y should be shape (batch, label sequence length)
        """
        y = self.embedding(y)
        # preprend zeros
        b, t, h = y.shape
        start = torch.zeros((b, 1, h))
        if self.is_cuda:
            start = start.cuda()
        y = torch.cat([start, y], dim=1)
        y, _ = self.dec_rnn(y)
        # Combine the input states and the output states
        x = x.unsqueeze(dim=2)
        y = y.unsqueeze(dim=1)

        x = self.fc1(x)
        y = self.fc1(y)
        z = self.fc1(z)

        z = z.unsqueeze(dim = 1)

        out = x + y + z
        out = nn.functional.relu(out)
        out = self.fc2(out)
        out = nn.functional.log_softmax(out, dim=3)
        return out

    def collate(self, inputs, labels):
        example = inputs[0]

        max_t = max(i.shape[0] for i in inputs)
        max_t = self.conv_out_size(max_t, 0)
        x_lens = torch.IntTensor([max_t] * len(inputs))
        x = torch.FloatTensor(model.zero_pad_concat(inputs))
        y_lens = torch.IntTensor([len(l) for l in labels])
        y = torch.IntTensor([l for label in labels for l in label])
        batch = [x, y, x_lens, y_lens]
        if self.volatile:
            for v in batch:
                v.volatile = True
        return batch

    def collate_bias(self, bias):
        max_t = max(i.shape[0] for i in bias)
        max_t = self.conv_out_size(max_t, 0)
        z_lens = torch.IntTensor([max_t] * len(bias))
        z = torch.FloatTensor(model.zero_pad_concat(bias))
        batch = [z, z_lens]
        return batch


    def infer(self, batch, beam_size=4):
        out = self(batch)
        out = out.cpu().data.numpy()
        preds = []
        for e, (i, l) in enumerate(zip(*batch)):
            T = i.shape[0]
            U = len(l) + 1
            lp = out[e, :T, :U, :]
            preds.append(td.decode_static(lp, beam_size, blank=self.blank)[0])
        return preds

    def label_collate(self, labels):
        # Doesn't matter what we pad the end with
        # since it will be ignored.
        batch_size = len(labels)
        end_tok = labels[0][-1]
        max_len = max(len(l) for l in labels)
        cat_labels = np.full((batch_size, max_len),
                        fill_value=end_tok, dtype=np.int64)
        for e, l in enumerate(labels):
            cat_labels[e, :len(l)] = l
        labels = torch.LongTensor(cat_labels)
        if self.volatile:
            labels.volatile = True
        return labels

    def preprocess(self, filename):
        listOfContext = []
        ord_a = 97
        with open(filename) as f:
            for line in f:
                listOfContext.append(json.loads(line)["text"])
        listOfContext = list(filter(None,listOfContext))
        listOfTensors = []
        numInput = len(listOfContext)
        for inp in listOfContext:
            inpLen = len(inp)
            inpTensor = torch.zeros(inpLen, 32) #inorder to do 8
            for li, letter in enumerate(inp):
                inpTensor[li][ord(letter)-ord_a] = 1
            listOfTensors.append(inpTensor)
        return listOfTensors, numInput

    def zero_pad_concat_bias(self, inputs):
        max_t = max(inp.shape[0] for inp in inputs)
        shape = (len(inputs), max_t, inputs[0].shape[1])
        input_mat = np.zeros(shape, dtype=np.float32)
        for e, inp in enumerate(inputs):
            input_mat[e, :inp.shape[0], :] = inp
        return input_mat

class biasEncoder(nn.Module):

    def __init__(self, input_size, output_size, embedding_dim, bias_cnn_dim):
        super(biasEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 256)
        self.cnn = torch.nn.Conv1d(in_channels = bias_cnn_dim, out_channels = 1, kernel_size = 2, stride = 1)

    def forward(self, z):
        z = z.long()
        z = self.embed(z)
        lstm_out, _ = self.lstm(z.view(len(z),-1,self.embedding_dim).contiguous())
        smaller_out = self.cnn(lstm_out)
        z_mat = torch.zeros([8,1,256])
        z_mat[:,:,:255] = smaller_out
        return z_mat
