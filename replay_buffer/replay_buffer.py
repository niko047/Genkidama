import torch
import torch.multiprocessing as mp

class ReplayBuffers(object):

    def __init__(self, shared_replay_buffer, cpu_id, max_num_interactions, batch_size):
        self.shared_replay_buffer = shared_replay_buffer
        self.cpu_id = cpu_id
        self.current_iter_number = 0
        self.max_num_interactions = max_num_interactions

        self.has_been_filled = False
        self.batch_size = batch_size

    @staticmethod
    def init_global_buffer(len_interaction: int,
                           num_iters: int,
                           tot_num_cpus: int,
                           dtype: torch.dtype) -> torch.Tensor:
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
            # Uses a First-In First-Out way of rewriting the buffer by resetting index
            self.current_iter_number = 0

            # Setting it to true, it is going to help indexing while random sampling
            self.has_been_filled = True

        self.shared_replay_buffer[self.cpu_id, self.current_iter_number, :] = new_inter

        self.current_iter_number += 1

    def is_buffer_full(self):
        """Checks if the buffer for current CPU is full"""
        return True if (self.max_num_interactions - self.current_iter_number + 1) == 0 else False

    @staticmethod
    def random_sample_batch_(shared_buffer: torch.Tensor,
                             len_interaction: int,
                             num_iters: int,
                             tot_num_cpus: int,
                             batch_size: int,
                             replacement=False) -> torch.Tensor:
        """Returns a random sample over the shared CPU buffers, mixing them up"""
        reshaped_shared_buffer = torch.reshape(shared_buffer, (num_iters*tot_num_cpus, len_interaction))
        
        # Creates boolean mask to exclude rows that are still zero
        zero_row = torch.zeros(size=(len_interaction,))
        mask = ~(reshaped_shared_buffer == zero_row)[:, 0]

        # Masks the array for valid rows
        masked_buffer = reshaped_shared_buffer[mask]

        # Random samples the data
        if replacement:
            # Random sample without fear of full of zeros-rows
            idxs = torch.randint(len(masked_buffer), (batch_size,))
        else:
            idxs = torch.randperm(len(masked_buffer))[:batch_size]

        return masked_buffer[idxs]

    def random_sample_batch(self, replacement=False):
        """Random samples a batch from shared buffer to update the network's weights"""
        max_idx = len(self.shared_replay_buffer[0]) if self.has_been_filled else self.current_iter_number + 1

        if replacement:
            # Random sample without fear of full of zeros-rows
            idxs = torch.randint(max_idx, (self.batch_size,))
        else:
            idxs = torch.randperm(max_idx)[:self.batch_size]

        return self.shared_replay_buffer[self.cpu_id][idxs]


