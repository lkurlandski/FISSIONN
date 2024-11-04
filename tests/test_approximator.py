"""
"""

from pprint import pformat
import unittest

import torch
from torch.nn.utils.rnn import pad_sequence

from src.approximator import *

import unittest
import torch


class TestApproximators(unittest.TestCase):

    batch_size = 5
    max_length = 71
    input_length = 59
    target_length = 61
    hidden_size = 60
    num_layers = 2
    nhead = 2
    intermediate_size = 80

    def setUp(self):

        self.rec = RecurrentApproximator(
            max_length=self.max_length,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            cell="rnn"
        )
        self.trn = TransformerApproximator(
            max_length=self.max_length,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            nhead=self.nhead,
            intermediate_size=self.intermediate_size,
        )

        self.inputs = ApproximatorCollateFn.add_special_tokens(torch.rand(self.batch_size, self.input_length - 2))
        self.targets = ApproximatorCollateFn.add_special_tokens(torch.rand(self.batch_size, self.target_length - 2))
        self.embeddings = torch.rand(self.batch_size, self.input_length, self.hidden_size)
        self.encoder_outputs = torch.rand(self.batch_size, self.input_length, self.hidden_size)
        self.encoder_hidden = torch.rand(self.num_layers, self.batch_size, self.hidden_size)
        self.decoder_outputs = torch.rand(self.batch_size, self.target_length, self.hidden_size)

    def _test_embed_src(self, embeddings: Tensor):
        assert embeddings.dim() == 3
        assert embeddings.size(0) == self.batch_size, f"Got shape={tuple(embeddings.shape)}. Expected shape={(self.batch_size, self.input_length, self.hidden_size)}."
        assert embeddings.size(1) == self.input_length, f"Got shape={tuple(embeddings.shape)}. Expected shape={(self.batch_size, self.input_length, self.hidden_size)}."
        assert embeddings.size(2) == self.hidden_size, f"Got shape={tuple(embeddings.shape)}. Expected shape={(self.batch_size, self.input_length, self.hidden_size)}."

    def test_embed_src_rec(self):
        embeddings = self.rec.embed_src(self.inputs)
        self._test_embed_src(embeddings)

    def test_embed_src_trn(self):
        embeddings = self.trn.embed_src(self.inputs)
        self._test_embed_src(embeddings)

    def _test_embed_tgt(self, embeddings: Tensor):
        assert embeddings.dim() == 3
        assert embeddings.size(0) == self.batch_size, f"Got shape={tuple(embeddings.shape)}. Expected shape={(self.batch_size, self.target_length, self.hidden_size)}."
        assert embeddings.size(1) == self.target_length, f"Got shape={tuple(embeddings.shape)}. Expected shape={(self.batch_size, self.target_length, self.hidden_size)}."
        assert embeddings.size(2) == self.hidden_size, f"Got shape={tuple(embeddings.shape)}. Expected shape={(self.batch_size, self.target_length, self.hidden_size)}."

    def test_embed_tgt_rec(self):
        embeddings = self.rec.embed_tgt(self.targets)
        self._test_embed_tgt(embeddings)

    def test_embed_tgt_trn(self):
        embeddings = self.trn.embed_tgt(self.targets)
        self._test_embed_tgt(embeddings)

    def _test_encoder_outputs(self, outputs: Tensor, hidden: Tensor):
        assert outputs.dim() == 3
        assert outputs.size(0) == self.batch_size
        assert outputs.size(1) == self.input_length
        assert outputs.size(2) == self.hidden_size

        if hidden is not None:
            assert hidden.dim() == 3
            assert hidden.size(0) == self.num_layers
            assert hidden.size(1) == self.batch_size
            assert hidden.size(2) == self.hidden_size

    def test_encode_rec(self):
        outputs, hidden = self.rec.encode(self.embeddings)
        self._test_encoder_outputs(outputs, hidden)

    def test_encode_trn(self):
        outputs = self.trn.encode(self.embeddings, None, None)
        self._test_encoder_outputs(outputs, None)

    def _test_decode(self, predictions: Tensor):
        assert predictions.dim() == 2
        assert predictions.size(0) == self.batch_size
        assert predictions.size(1) in (self.max_length, self.target_length), f"Got {tuple(predictions.shape)}. Expected ({self.batch_size}, {self.max_length}) or ({self.batch_size}, {self.target_length})."
        assert torch.all(predictions[:,0] == BOS), f"Got {predictions[:,0]}. Expected {BOS}."
        assert torch.all((predictions[:,-1] == EOS) | (predictions[:,-1] == PAD)), f"Got {predictions[:,-1]}. Expected {EOS} or {PAD}."

    def test_decode_rec_1(self):
        predictions = self.rec.decode(self.encoder_outputs, self.encoder_hidden, self.targets, 1.0)[0]
        self._test_decode(predictions)

    def test_decode_trn_1(self):
        predictions = self.trn.decode(self.encoder_outputs, self.targets, 1.0)
        self._test_decode(predictions)

    def test_decode_rec_2(self):
        predictions = self.rec.decode(self.encoder_outputs, self.encoder_hidden, None, 0.0)[0]
        self._test_decode(predictions)

    def test_decode_trn_2(self):
        predictions = self.trn.decode(self.encoder_outputs, None, 0.0)
        self._test_decode(predictions)

    def test_decode_rec_3(self):
        predictions = self.rec.decode(self.encoder_outputs, self.encoder_hidden, self.targets, 0.5)[0]
        self._test_decode(predictions)

    def test_decode_trn_3(self):
        predictions = self.trn.decode(self.encoder_outputs, self.targets, 0.5)
        self._test_decode(predictions)

    def _test_project(self, predictions: Tensor):
        assert predictions.dim() == 2
        assert predictions.size(0) == self.batch_size
        assert predictions.size(1) == self.target_length

    def test_project_rec(self):
        predictions = self.rec.project(self.decoder_outputs)
        self._test_project(predictions)

    def test_project_trn(self):
        predictions = self.trn.project(self.decoder_outputs)
        self._test_project(predictions)


if __name__ == "__main__":
    unittest.main()
