import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from buffer import ReplayBuffers
from manager import Manager
from shared_optims import SharedAdam
from neural_net import NNet

mp.set_start_method('spawn', force=True)

LEN_SINGLE_STATE = 2
LEN_ITERATIONS = 100
NUM_CPUS = mp.cpu_count()
NUM_EPISODES = 50
NUM_STEPS = 100

def train_model(glob_net, opt, buffer, i, semaphor, res_queue):
    loc_net = NNet()

    b = ReplayBuffers(shared_replay_buffer=buffer,
                      cpu_id=i,
                      len_interaction=LEN_SINGLE_STATE + 1,
                      batch_size=5,
                      num_iters=LEN_ITERATIONS,
                      tot_num_cpus=NUM_CPUS,
                      replacement=False)

    semaphor[b.cpu_id] = True

    while not torch.all(semaphor):
        pass

    for i in range(NUM_EPISODES):
        # Generate training data and update buffer
        for j in range(NUM_STEPS):
            tensor_tuple = Manager.data_generative_mechanism()
            b.record_interaction(tensor_tuple)

            # TODO - Uncomment for replay memory
            # A3C without replay memory does not need a replay buffer
            # The other algorithms do, we can implement an update consisting in a mini batch of 3 observations
            # sampled_batch = b.random_sample_batch(from_shared_memory=True) , set batch size inside

            if not ((j+1) % 5):
                sampled_batch = b.random_sample_batch(from_shared_memory=True)
                loc_output = loc_net.forward(sampled_batch[:, :-1])
                loss = F.mse_loss(loc_output, torch.Tensor(sampled_batch[:, -1]))
                res_queue.put(loss.mean().item())
                opt.zero_grad()
                loss.backward()

                # Perform the update of the global parameters using the local ones
                for lp, gp in zip(loc_net.parameters(), glob_net.parameters()):
                    gp._grad = lp.grad
                opt.step()

                loc_net.load_state_dict(glob_net.state_dict())
                print(f'EPISODE {i} STEP {j + 1} -> Loss for cpu {b.cpu_id} is: {loss}')

    res_queue.put(None)

if __name__ == '__main__':
    glob_net = NNet()
    glob_net.share_memory()
    opt = SharedAdam(glob_net.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer

    # Queue used to store the history of rewards while training
    res_queue = mp.Queue()

    # Initializes the global buffer, where interaction with the environment are stored
    buffer = ReplayBuffers.init_global_buffer(len_interaction=LEN_SINGLE_STATE + 1, # 2 inputs + 1 output
                                              num_iters=LEN_ITERATIONS,
                                              tot_num_cpus=NUM_CPUS,
                                              dtype=torch.float32)
    # Creates a starting semaphor
    semaphor = Manager.initialize_semaphor(NUM_CPUS)

    procs = [mp.Process(target=train_model, args=(glob_net, opt, buffer, i, semaphor, res_queue)) for i in range(mp.cpu_count())]

    [p.start() for p in procs]

    # Stuff to pop the rewards from the queue continuously until all agents are done and append None
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    [p.join() for p in procs]

    # Code for plotting the rewards

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Reward value')
    plt.xlabel('Step of the NN')
    plt.show()