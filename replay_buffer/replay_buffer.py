import torch
import torch.multiprocessing as mp
import numpy as np

mp.set_start_method('spawn', force=True)


class ReplayBuffers(object):

    def __init__(self, shared_replay_buffer, cpu_id, max_num_interactions):
        self.shared_replay_buffer = shared_replay_buffer
        self.cpu_id = cpu_id
        self.current_iter_number = 0
        self.max_num_interactions = max_num_interactions

    @staticmethod
    def init_global_buffer(len_interaction: int,
                           num_iters: int,
                           tot_num_cpus: int,
                           dtype: torch.dtype):
        # Creates a Replay buffer filled with zeroes
        shared_tensor_buffer = torch.zeros(
            size=(tot_num_cpus, num_iters, len_interaction),
            dtype=dtype
        )

        # Shares the memory of the tensor across processes
        shared_tensor_buffer.share_memory_()

        return shared_tensor_buffer

    def record_interaction(self, new_inter: torch.Tensor):
        """Records an interaction inside the shared replay buffer memory"""
        # Checks if the buffer is full
        if self.is_buffer_full():
            # TODO - Resets the buffer memory of the current worker
            pass
        self.shared_replay_buffer[self.cpu_id, self.current_iter_number, :] = new_inter

        self.current_iter_number += 1

    def is_buffer_full(self):
        """Checks if the buffer for current CPU is full"""
        return True if (self.max_num_interactions - self.current_iter_number + 1) == 0 else False

    def random_sample_batch(self):
        """Random samples a batch from shared buffer to update the network's weights"""
        pass


q = []

# import multiprocessing as mp
LEN_SINGLE_STATE = 4
LEN_ITERATIONS = 5
NUM_CPUS = mp.cpu_count()


def foo(t, i, max_num_interactions):
    b = ReplayBuffers(shared_replay_buffer=t, cpu_id=i, max_num_interactions=max_num_interactions)

    # Get the array from shared memory and reshape it into a numpy array
    for j in range(LEN_ITERATIONS):
        print(b.shared_replay_buffer)
        b.record_interaction(torch.Tensor([1, 2, 3, 4]))


if __name__ == '__main__':
    buffer = ReplayBuffers.init_global_buffer(len_interaction=LEN_SINGLE_STATE,
                                              num_iters=LEN_ITERATIONS,
                                              tot_num_cpus=mp.cpu_count(),
                                              dtype=torch.float32)
    print(f'Buffer is {buffer}')

    procs = [mp.Process(target=foo, args=(buffer, i, LEN_ITERATIONS,)) for i in range(mp.cpu_count())]
    [p.start() for p in procs]
    [p.join() for p in procs]

    print(f'New buffer is {buffer}')

    # Stacks the observations coming from all of them
    # reshaped_t = torch.reshape(t, (NUM_CPUS * LEN_ITERATIONS, LEN_SINGLE_STATE))
    # print(reshaped_t)
    # Should random sample without replacement all the items in the subset: shuffle rows
    # shuffled_t = reshaped_t[torch.randint(len(reshaped_t), (len(reshaped_t),))]
