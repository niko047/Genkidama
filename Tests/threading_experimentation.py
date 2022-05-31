import threading

class Counter:
    def __init__(self):
        self.gcounter = 0

    def add1(self):
        for i in range(100):
            self.gcounter += 1

if __name__ == '__main__':
    c = Counter()
    threads = [threading.Thread(target=c.add1) for j in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    print(f'Count is {c.gcounter}')