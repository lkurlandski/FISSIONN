"""
"""
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from src.finn import DynamicFINNDataset, StaticFINNDataset


class TestFINNDataset(unittest.TestCase):

    def setUp(self):
        self.flow_lengths = [np.random.randint(50, 101) for _ in range(10)]
        self.ipds = [np.random.rand(l) for l in self.flow_lengths]
        self.fingerprint_length = 32
        self.amplitude = 1
        self.noise_deviation_low = 0.1
        self.noise_deviation_high = 0.5

    def test_dynamic_finn_dataset(self):
        dataset = DynamicFINNDataset(
            ipds=self.ipds,
            fingerprint_length=self.fingerprint_length,
            amplitude=self.amplitude,
            noise_deviation_low=self.noise_deviation_low,
            noise_deviation_high=self.noise_deviation_high,
        )

        self.assertEqual(len(dataset), len(self.ipds))

        fingerprints, ipds, delays, noises = [], [], [], []
        for i in range(len(dataset)):
            fingerprint, ipd, delay, noise = dataset[i]
            self.assertEqual(fingerprint.shape, (self.fingerprint_length,))
            self.assertEqual(ipd.shape, (self.flow_lengths[i],))
            self.assertEqual(delay.shape, (self.flow_lengths[i],))
            self.assertEqual(noise.shape, (self.flow_lengths[i],))
            fingerprints.append(fingerprint)
            ipds.append(ipd)
            delays.append(delay)
            noises.append(noise)

        # The dynamic dataset should return different data each time its iterated.
        for i in range(len(dataset)):
            fingerprint, ipd, delay, noise = dataset[i]
            self.assertNotEqual(fingerprint.tolist(), fingerprints[i].tolist())
            self.assertEqual(ipd.tolist(), ipds[i].tolist())
            self.assertNotEqual(delay.tolist(), delays[i].tolist())
            self.assertNotEqual(noise.tolist(), noises[i].tolist())


    def test_static_finn_dataset(self):
        dataset = StaticFINNDataset(
            ipds=self.ipds,
            fingerprint_length=self.fingerprint_length,
            amplitude=self.amplitude,
            noise_deviation_low=self.noise_deviation_low,
            noise_deviation_high=self.noise_deviation_high,
        )

        self.assertEqual(len(dataset), len(self.ipds))

        fingerprints, ipds, delays, noises = [], [], [], []
        for i in range(len(dataset)):
            fingerprint, ipd, delay, noise = dataset[i]
            self.assertEqual(fingerprint.shape, (self.fingerprint_length,))
            self.assertEqual(ipd.shape, (self.flow_lengths[i],))
            self.assertEqual(delay.shape, (self.flow_lengths[i],))
            self.assertEqual(noise.shape, (self.flow_lengths[i],))
            fingerprints.append(fingerprint)
            ipds.append(ipd)
            delays.append(delay)
            noises.append(noise)

        # The static dataset should return the same data each time its iterated.
        for i in range(len(dataset)):
            fingerprint, ipd, delay, noise = dataset[i]
            self.assertEqual(fingerprint.tolist(), fingerprints[i].tolist())
            self.assertEqual(ipd.tolist(), ipds[i].tolist())
            self.assertEqual(delay.tolist(), delays[i].tolist())
            self.assertEqual(noise.tolist(), noises[i].tolist())


if __name__ == '__main__':
    unittest.main()
