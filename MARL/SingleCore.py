import torch.multiprocessing as mp


class SingleCore(mp.Process):
    def __init__(self):
        super(SingleCore, self).__init__()
