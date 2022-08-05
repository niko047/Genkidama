import threading
import torch.nn
import torch.nn as nn
import torch.nn.functional as F


class Counter:
    def __init__(self):
        self.gcounter = 0

    def add1(self):
        for i in range(100):
            self.gcounter += 1

    def run(self):
        for i in range(5):
            t = threading.Thread(target=self.add1, args=())
            t.start()
            t.join()


if __name__ == '__main__':
    c = Counter()
    c.run()
    print(f'Count is {c.gcounter}')