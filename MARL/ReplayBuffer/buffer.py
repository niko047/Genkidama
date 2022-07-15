import torch


class ReplayBuffers(object):

    def __init__(self, shared_replay_buffer: torch.Tensor,
                 cpu_id: int,
                 len_interaction: int,
                 num_iters: int,
                 batch_size: int,
                 tot_num_cpus: int,
                 replacement: bool,
                 sample_from_shared_memory: bool,
                 time_ordered_sampling: bool,
                 len_state: int):
        """
        :param shared_replay_buffer:    Tensor shared amongst CPU processes
        :param cpu_id:                  Unique id of the CPU using the current object
        :param len_interaction:         len(X_i) + len(y_i), features + output length at any given state s_i (constant)
        :param num_iters:               Number of iterations after which the replay memory for CPU_{i} is full
        :param batch_size:              Batch size of the sampled batch for updating the net
        :param tot_num_cpus:            Total number of CPUs available within this same machine
        :param replacement:             Sampling method from replay memory
        """
        self.shared_replay_buffer = shared_replay_buffer
        self.cpu_id = cpu_id
        self.current_iter_number = 0
        self.len_interaction = len_interaction
        self.num_iters = num_iters
        self.tot_num_cpus = tot_num_cpus
        self.replacement = replacement
        self.batch_size = batch_size
        self.sample_from_shared_memory = sample_from_shared_memory
        self.time_ordered_sampling = time_ordered_sampling

        self.len_state = len_state
        self.len_action = 1
        self.len_reward = 1

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
        # print(f'Trying to insert at index {self.current_iter_number} the iteration {new_inter}')
        # print(f'As of now the buffer for this cpu is {self.shared_replay_buffer[self.cpu_id, :, :]}')
        self.shared_replay_buffer[self.cpu_id, self.current_iter_number, :] = new_inter

        self.current_iter_number += 1

    def is_buffer_full(self) -> bool:
        """Checks if the buffer for current CPU is full"""
        # print(f'Current length of interaction is {self.len_interaction} should be 5')
        # print(f'Current iteration number + 1 is {self.current_iter_number + 1}')
        return True if (self.num_iters - self.current_iter_number) == 0 else False

    @staticmethod
    def random_sample_batch_(shared_buffer: torch.Tensor,
                             len_interaction: int,
                             len_state: int,
                             len_action: int,
                             len_reward: int,
                             num_iters: int,
                             tot_num_cpus: int,
                             batch_size: int,
                             replacement: bool = False,
                             time_ordered_sampling=True,
                             cpu_id=None) :
        """Returns a random sample over the shared CPU buffers, mixing them up"""
        if cpu_id is None:
            reshaped_shared_buffer = torch.reshape(shared_buffer, (num_iters * tot_num_cpus, len_interaction))
        elif isinstance(cpu_id, int):
            reshaped_shared_buffer = shared_buffer[cpu_id]
        else:
            raise Exception('cpu_id does not conform to type None or int')

        # Creates boolean mask to exclude rows that are still zero
        zero_row = torch.zeros(size=(len_interaction,))
        mask = ~(reshaped_shared_buffer == zero_row)[:, 0]
        # Masks the array for valid rows
        masked_buffer = reshaped_shared_buffer[mask]

        # Sampling should NOT follow the timestamps of the trajectory recorded
        if not time_ordered_sampling:

            # Random samples the data
            if replacement:
                # Random sample without fear of full of zeros-rows
                idxs = torch.randint(len(masked_buffer), (batch_size,))
            else:
                idxs = torch.randperm(len(masked_buffer))[:batch_size]

            return masked_buffer[idxs, : len_state], \
                   masked_buffer[idxs, len_state: len_state + len_action], \
                   masked_buffer[idxs, len_state + len_action:]

        # Sampling should follow the timestamps of the trajectory recorded in the buffer, returns whole buffer
        else:
            return masked_buffer[:, : len_state], \
                   masked_buffer[:, len_state: len_state + len_action], \
                   masked_buffer[:, len_state + len_action:]


    def random_sample_batch(self) -> torch.Tensor:
        """Random samples a batch from shared buffer to update the network's weights"""
        state, action, reward = self.random_sample_batch_(shared_buffer=self.shared_replay_buffer,
                                                 len_interaction=self.len_interaction,
                                                 len_state=self.len_state,
                                                 len_action=self.len_action,
                                                 len_reward=self.len_reward,
                                                 num_iters=self.num_iters,
                                                 tot_num_cpus=self.tot_num_cpus,
                                                 batch_size=self.batch_size,
                                                 replacement=self.replacement,
                                                 time_ordered_sampling=self.time_ordered_sampling,
                                                 cpu_id=None if self.sample_from_shared_memory else self.cpu_id)
        return state, action, reward