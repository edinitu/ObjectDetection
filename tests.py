import time
import unittest
import training
import torch
import utils


class TestMethods(unittest.TestCase):

    def test_loss(self):
        outputs = torch.rand((64, 2058))
        truth = torch.rand((64, 2058))
        for i in range(64):
            for j in range(0, 2058, 42):
                truth[i, j] = 1
        result = training.loss_calc(outputs, truth)
        # check that loss can't take very large values
        self.assertTrue(result.item() < 500)

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

    def test_plot(self):
        d = utils.DynamicUpdate('test')
        utils.plot_dynamic_graph(d, 1, 1)
        time.sleep(2)
        utils.plot_dynamic_graph(d, 1, 2)
        time.sleep(2)
        utils.plot_dynamic_graph(d, 2, 3)
        time.sleep(2)

