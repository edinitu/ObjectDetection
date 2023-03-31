from sklearn.metrics import auc
import numpy as np

TRUE_POSITIVE = 'TP'
FALSE_POSITIVE = 'FP'


class PredictionStats:

    def __init__(self, confidence, confusion):
        self.confidence = confidence
        self.confusion = confusion

    def __str__(self):
        return str(f'Confidence: {self.confidence} - {self.confusion}')

    def __repr__(self):
        return str(f'Confidence: {self.confidence} - {self.confusion}')

    @classmethod
    def no_args_construct(cls):
        return cls(0, '')

    def set_tp(self):
        self.confusion = TRUE_POSITIVE

    def set_fp(self):
        self.confusion = FALSE_POSITIVE

    def get_confusion(self):
        return self.confusion

    def set_confidence(self, confidence):
        self.confidence = confidence

    def get_confidence(self):
        return self.confidence


class AveragePrecision:

    def __init__(self, prediction_stats_list, positives_count):
        self.average_precision = 0
        self.positives_count = positives_count
        self.predictions = prediction_stats_list
        self.compute_metrics()

    def compute_metrics(self):
        sorted_conf = []
        for elem in self.predictions:
            sorted_conf.append(elem.get_confidence())
        sorted_conf.sort(reverse=True)
        sorted_predictions = []
        for conf in sorted_conf:
            for elem in self.predictions:
                if elem.get_confidence() == conf:
                    sorted_predictions.append(elem)
                    break

        guessed = 0
        predicted = 0
        idx = 0
        recalls = []
        precisions = []
        recalls.append(0)
        for elem in sorted_predictions:
            if elem.get_confusion() == TRUE_POSITIVE:
                guessed += 1
                predicted += 1
            elif elem.get_confusion() == FALSE_POSITIVE:
                predicted += 1
            try:
                recalls.append(guessed/self.positives_count)
                precisions.append(guessed/predicted)
            except ZeroDivisionError:
                recalls.append(-1)

            if idx == 0:
                precisions.append(guessed/predicted)
                idx += 1

        if len(precisions) == 0:
            self.average_precision = 0
        else:
            self.average_precision = auc(np.asarray(recalls), np.asarray(precisions))

    def get_average_precision(self):
        return float(self.average_precision)
