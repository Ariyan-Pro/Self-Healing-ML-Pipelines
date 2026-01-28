class Rule:
    def __init__(self, metric, threshold, action):
        self.metric = metric
        self.threshold = threshold
        self.action = action

    def evaluate(self, signals):
        return signals.get(self.metric, 0) > self.threshold
