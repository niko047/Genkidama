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


def foo(t, i, max_num_interactions):
    b = ReplayBuffers(shared_replay_buffer=t, cpu_id=i, len_interaction=LEN_SINGLE_STATE, batch_size=3, num_iters=LEN_ITERATIONS,
                      tot_num_cpus = NUM_CPUS, replacement = False)

    # Get the array from shared memory and reshape it into a numpy array
    for j in range(LEN_ITERATIONS):
        b.record_interaction(torch.Tensor([i]*4))

    print(f'Random sampled batch is {b.random_sample_batch(from_shared_memory=False)}')

def task_handler(cpu_id):
    if cpu_id == 0:
        # Behave as the central node handling updates
        pass
    else:
        # Behave as a working node
        pass

# TODO - Define a class Manager
# TODO - Define a Maneger.task_handler
# TODO - Define a Manager.semaphor

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
    print(f'Buffer is {buffer}')

    procs = [mp.Process(target=foo, args=(buffer, i, LEN_ITERATIONS,)) for i in range(mp.cpu_count())]
    [p.start() for p in procs]
    [p.join() for p in procs]

    print(f'New buffer is {buffer}')
    sample = ReplayBuffers.random_sample_batch_(shared_buffer = buffer,
                             len_interaction = LEN_SINGLE_STATE,
                             num_iters = LEN_ITERATIONS,
                             tot_num_cpus = NUM_CPUS,
                             batch_size = 3,
                             replacement=False)

    print(f'buffer is {buffer} shared memory {buffer.is_shared()}')