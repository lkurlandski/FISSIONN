"""
"""

from pprint import pformat
import unittest

import torch
from torch.nn.utils.rnn import pad_sequence

from src.approximator import *


class TestDataset(unittest.TestCase):

    def ipds_to_str(self, ipds: np.ndarray) -> str:
        return pformat([f"{ipd:.2e}" for ipd in ipds])

    def test_get_synthetic_sample(self):
        for _ in range(100):
            ipds = ApproximatorDataset.get_synthetic_sample()

    def test_get_synthetic_hop(self):
        runs = 100
        for _ in range(runs):
            ipds_a = ApproximatorDataset.get_synthetic_sample()
            ipds_b = ApproximatorDataset.get_synthetic_hop(ipds_a)


class TestTranslate(unittest.TestCase):

    batch_size = 4
    max_length = 256

    def setUp(self):
        x = [ApproximatorDataset.get_synthetic_sample() for _ in range(self.batch_size)]
        x = [torch.from_numpy(x_i).to(torch.float32) for x_i in x]
        x = pad_sequence(x, batch_first=True)
        x = x[:,:self.max_length - 2]
        b = bos((self.batch_size, 1))
        e = eos((self.batch_size, 1))
        self.x = torch.cat([b, x, e], dim=1)
        self.model = TransformerApproximator(self.max_length, **TransformerApproximator.SMALL)

    def test_greedy(self):
        y = self.model.translate(self.x, self.max_length, "greedy")
        print(y)

    unittest.skip("Skipping beam test")
    def test_beam(self):
        y = self.model.translate(self.x, self.max_length, "beam", num_beams=4)
        print(y)


if __name__ == "__main__":
    unittest.main()
