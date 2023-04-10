from enum import Enum


class ProofTransferMethod(Enum):
    REUSE = 1
    ALL = 3


class IVAN:
    def __init__(self, alpha, threshold):
        self.alpha = alpha
        self.threshold = threshold

    def __str__(self):
        return 'PRUNE('+str(round(self.alpha, 3)) + ',' + str(round(self.threshold, 3)) + ')'


class REORDERING:
    def __init__(self, alpha, threshold):
        self.alpha = alpha
        self.threshold = threshold

    def __str__(self):
        return 'REORDER('+str(round(self.alpha, 3)) + ',' + str(round(self.threshold, 3)) + ')'
