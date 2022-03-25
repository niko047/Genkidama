from buffer import ReplayBuffers
import torch
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

LEN_SINGLE_STATE = 4
LEN_ITERATIONS = 5
NUM_CPUS = mp.cpu_count()

class Manager(object):
    def __init__(self):
        self.a = 5

    @staticmethod
    def initialize_semaphor(num_workers):
        """Shared semaphor that when filled with 1's, allows network updates to begin"""
        s = torch.Tensor([0]*num_workers).to(torch.bool)
        s.share_memory_()
        return s
    
    @staticmethod
    def initialize_queue(len_queue):
        """Initializes a first-in first-out queue of batches updates of the shared network"""
        return mp.Queue(len_queue)


def foo(t, i, semaphor, queue):
    b = ReplayBuffers(shared_replay_buffer=t, cpu_id=i, len_interaction=LEN_SINGLE_STATE, batch_size=3, num_iters=LEN_ITERATIONS,
                      tot_num_cpus = NUM_CPUS, replacement=False)

    # Get the array from shared memory and reshape it into a numpy array
    for j in range(LEN_ITERATIONS):
        b.record_interaction(torch.Tensor([i]*4))
    # Green light
    semaphor[b.cpu_id] = True

    while not torch.all(semaphor):
        pass

    # Insert random sampled batch into queue for the Net's updates
    queue.put(b.random_sample_batch(from_shared_memory=False))


def task_handler(cpu_id):
    if cpu_id == 0:
        # Behave as the central node handling updates
        pass
    else:
        # Behave as a working node
        pass

# TODO - Implement a manager class, it should start random sampling once all CPUs have reached a certain number
# TODO - of steps in their environment. Then they can all go freely. After that, Implement the result of the random sampling
# TODO - in such a way that the random sampled tensors go inside a multiprocessing queue and get ready for updating the network
# TODO - Every X interactions with the environment, the workers check if the queue is busy, and if so take one bacth and
# TODO - perform one update to the global parameters of the network.


if __name__ == '__main__':
    buffer = ReplayBuffers.init_global_buffer(len_interaction=LEN_SINGLE_STATE,
                                              num_iters=LEN_ITERATIONS,
                                              tot_num_cpus=NUM_CPUS,
                                              dtype=torch.float32)
    semaphor = Manager.initialize_semaphor(NUM_CPUS)
    queue = Manager.initialize_queue(NUM_CPUS)

    print(f'Buffer is {buffer}')

    procs = [mp.Process(target=foo, args=(buffer, i, semaphor, queue)) for i in range(mp.cpu_count())]
    [p.start() for p in procs]
    [p.join() for p in procs]

    for j in range(NUM_CPUS):
        print(queue.get(j))