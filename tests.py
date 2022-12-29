import unittest
import training
import torch


class TestMethods(unittest.TestCase):

    def test_loss(self):
        outputs = torch.ones((64, 2058))
        truth = torch.zeros((64, 2058))
        result = training.loss_calc(outputs, truth)
        self.assertEqual(result, torch.ones(1, requires_grad=True))
