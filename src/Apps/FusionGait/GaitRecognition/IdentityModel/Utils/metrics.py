class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, value, n=1):
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(pred, target):
    """
    pred: tensor (batch,)
    target: tensor (batch,)
    """
    correct = (pred == target).sum().item()
    total = target.size(0)
    return correct / total
