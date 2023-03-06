class PredictionStats:

    def __init__(self):
        self.confidence = 0
        self.confusion = ''

    def set_tp(self):
        self.confusion = 'TP'

    def set_fp(self):
        self.confusion = 'FP'

    def get_confusion(self):
        return self.confusion

    def set_confidence(self, confidence):
        self.confidence = confidence

    def get_confidence(self):
        return self.confidence


class AveragePrecision:

    def __init__(self, prediction_stats_list):
        self.precisions = []
        self.recalls = []
        self.predictions = prediction_stats_list

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
        # TODO Compute precision, recall, area under curve




