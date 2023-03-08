import time
import unittest
import training
from convertToYOLOcsv import convertToYOLO
import torch
import utils
from metrics import PredictionStats, AveragePrecision, TRUE_POSITIVE, FALSE_POSITIVE


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

    def test_yolo_conversion(self):
        bbox_initial_points = [100, 100, 400, 100, 400, 400, 100, 400]
        bbox_yolo = convertToYOLO(bbox_initial_points)
        bbox_reverse_conversion = utils.conv_yolo_2_dota(bbox_yolo)
        self.assertEqual(bbox_reverse_conversion, bbox_initial_points)

    def test_average_precision(self):
        image_predictions = [
            PredictionStats(0.9, TRUE_POSITIVE),
            PredictionStats(0.8, FALSE_POSITIVE),
            PredictionStats(0.7, TRUE_POSITIVE),
            PredictionStats(0.6, FALSE_POSITIVE),
            PredictionStats(0.5, TRUE_POSITIVE),
            PredictionStats(0.3, FALSE_POSITIVE),
            PredictionStats(0.2, FALSE_POSITIVE)
        ]
        avg = AveragePrecision(image_predictions, 4)
        self.assertEqual(0.533, round(avg.get_average_precision(), 3))

        image_predictions[0] = PredictionStats(0.9, FALSE_POSITIVE)
        avg = AveragePrecision(image_predictions, 4)
        self.assertEqual(0.123, round(avg.get_average_precision(), 3))

    def test_detection_list(self):
        outputs = torch.zeros((49*6), dtype=torch.float16)
        outputs[0] = 0.78
        outputs[1] = 0.7
        outputs[2] = 0.5
        outputs[3] = 0.5
        outputs[4] = 0.6

        truth = torch.zeros((49 * 6), dtype=torch.float16)
        truth[0] = 1
        truth[1] = 0.7
        truth[2] = 0.5
        truth[3] = 0.6
        truth[4] = 0.5

        iou = utils.get_iou([0.7, 0.5, 0.5, 0.6], [0.7, 0.5, 0.6, 0.5])
        self.assertTrue(iou > 0.5)

        utils.FinalPredictions(outputs, truth)

        outputs = torch.zeros((49*6), dtype=torch.float16)
        outputs[0] = 0.67
        outputs[1] = 0.3
        outputs[2] = 0.5
        outputs[3] = 0.5
        outputs[4] = 0.6

        truth = torch.zeros((49 * 6), dtype=torch.float16)
        truth[0] = 1
        truth[1] = 0.7
        truth[2] = 0.5
        truth[3] = 0.6
        truth[4] = 0.5

        iou = utils.get_iou([0.3, 0.5, 0.5, 0.6], [0.7, 0.5, 0.6, 0.5])
        self.assertTrue(iou < 0.5)

        utils.FinalPredictions(outputs, truth)

        self.assertTrue(
            utils.all_detections[0].get_confusion() == utils.TRUE_POSITIVE and
            utils.all_detections[1].get_confusion() == utils.FALSE_POSITIVE and
            utils.positives == 2
        )

        ap = AveragePrecision(utils.all_detections, utils.positives)
        self.assertTrue(ap.get_average_precision() == 0.5)


