from replay_buffer.replay_buffer import ReplayBuffers
import torch
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

LEN_SINGLE_STATE = 4
LEN_ITERATIONS = 5
NUM_CPUS = mp.cpu_count()

def foo(t, i, max_num_interactions):
    b = ReplayBuffers(shared_replay_buffer=t, cpu_id=i, max_num_interactions=max_num_interactions, batch_size=3)

    # Get the array from shared memory and reshape it into a numpy array
    for j in range(LEN_ITERATIONS):
        b.record_interaction(torch.Tensor([i]*4))


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
    print(f'Random sampled array is {sample}, memory shared {sample.is_shared()}')

    print(f'buffer is {buffer} shared memory {buffer.is_shared()}')