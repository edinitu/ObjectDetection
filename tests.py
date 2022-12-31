import unittest
import training
import torch
import utils


class TestMethods(unittest.TestCase):

    def test_loss(self):
        outputs = torch.ones((64, 2058))
        truth = torch.zeros((64, 2058))
        result = training.loss_calc(outputs, truth)
        self.assertEqual(result, torch.ones(1, requires_grad=True))

    def test_iou(self):
        # no overlap
        bbox1 = (0.5, 0.5, 0.1, 0.1)
        bbox2 = (0.1, 0.1, 0.01, 0.01)
        expected_iou = 0
        actual_iou = utils.get_iou(bbox1, bbox2)
        self.assertEqual(expected_iou, actual_iou)

        # full overlap
        bbox1 = (0.5, 0.5, 0.1, 0.1)
        bbox2 = (0.5, 0.5, 0.1, 0.1)
        expected_iou = 1
        actual_iou = utils.get_iou(bbox1, bbox2)
        self.assertEqual(expected_iou, round(actual_iou))

        # partial overlap
        bbox1 = (0.5, 0.5, 0.2, 0.1)
        bbox2 = (0.4, 0.4, 0.2, 0.2)
        expected_iou = 0.2
        actual_iou = round(utils.get_iou(bbox1, bbox2), 1)
        self.assertEqual(expected_iou, actual_iou)
