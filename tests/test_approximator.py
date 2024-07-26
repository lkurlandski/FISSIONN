"""
"""

from pprint import pformat
import unittest

import torch
from torch.nn.utils.rnn import pad_sequence

from src.approximator import *


class TestRecurrentApproximator(unittest.TestCase):

    batch_size = 5
    max_length = 71
    hidden_size = 60
    num_layers = 2

    def setUp(self):

        self.model = RecurrentApproximator(
            max_length=self.max_length,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            cell="rnn"
        )
        self.inputs = torch.rand(self.batch_size, 59)
        self.targets = torch.rand(self.batch_size, 61)
        self.embeddings = torch.rand(self.batch_size, 59, self.hidden_size)
        self.encoder_outputs = torch.rand(self.batch_size, 59, self.hidden_size)
        self.encoder_hidden = torch.rand(self.num_layers, self.batch_size, self.hidden_size)
        self.decoder_outputs = torch.rand(self.batch_size, 61, self.hidden_size)

    def test_embed(self):
        embeddings = self.model.embed(self.inputs)
        assert embeddings.dim() == 3

    def test_encode(self):
        outputs, hidden = self.model.encode(self.embeddings)
        assert outputs.dim() == 3
        assert hidden.dim() == 3

    def test_decode_1(self):
        predictions = self.model.decode(self.encoder_outputs, self.encoder_hidden, self.targets, ratio=1.0)[0]
        assert predictions.dim() == 2

    def test_decode_2(self):
        predictions = self.model.decode(self.encoder_outputs, self.encoder_hidden, None, ratio=0.0)[0]
        assert predictions.dim() == 2

    def test_decode_3(self):
        predictions = self.model.decode(self.encoder_outputs, self.encoder_hidden, self.targets, ratio=0.5)[0]
        assert predictions.dim() == 2

    def test_project(self):
        predictions = self.model.project(self.decoder_outputs)
        assert predictions.dim() == 2


class TestTransformerApproximator(unittest.TestCase):

    batch_size = 5
    max_length = 71
    hidden_size = 60
    num_layers = 3
    nhead = 2
    intermediate_size = 80

    def setUp(self):

        self.model = TransformerApproximator(
            max_length=self.max_length,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            nhead=self.nhead,
            intermediate_size=self.intermediate_size,
        )
        self.inputs = torch.rand(self.batch_size, 59)
        self.targets = torch.rand(self.batch_size, 61)
        self.embeddings = torch.rand(self.batch_size, 59, self.hidden_size)
        self.encoder_outputs = torch.rand(self.batch_size, 59, self.hidden_size)
        self.encoder_hidden = torch.rand(self.num_layers, self.batch_size, self.hidden_size)
        self.decoder_outputs = torch.rand(self.batch_size, 61, self.hidden_size)

    def test_embed(self):
        embeddings = self.model.embed(self.inputs)
        assert embeddings.dim() == 3

    def test_encode(self):
        encoder_outputs = self.model.encode(self.embeddings, None, None)
        assert encoder_outputs.dim() == 3

    def test_decode_1(self):
        predictions = self.model.decode(self.encoder_outputs, self.targets, ratio=1.0)
        assert predictions.dim() == 2

    def test_decode_2(self):
        predictions = self.model.decode(self.encoder_outputs, None, ratio=1.0)
        assert predictions.dim() == 2

    def test_decode_3(self):
        predictions = self.model.decode(self.encoder_outputs, self.targets, ratio=0.5)
        assert predictions.dim() == 2

    def test_project(self):
        predictions = self.model.project(self.decoder_outputs)
        assert predictions.dim() == 2


if __name__ == "__main__":
    unittest.main()
