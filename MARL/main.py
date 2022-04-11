import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from buffer import ReplayBuffers
from manager import Manager
from shared_optims import SharedAdam
from Nets.neural_net import ToyNet

mp.set_start_method('spawn', force=True)

LEN_INPUTS_X: int = 2
LEN_OUTPUTS_Y: int = 1
LEN_ITERATIONS: int = 50
NUM_CPUS: int = mp.cpu_count()
NUM_EPISODES: int = 60
NUM_STEPS: int = 200
BATCH_SIZE: int = 5
SAMPLE_FROM_SHARED_MEMORY: bool = True
SAMPLE_WITH_REPLACEMENT: bool = False


""" 
TO-DO
1. Obtain the parameters
2. Flatten them all with a = torch.nn.utils.parameters_to_vector(glob_net.parameters())
3. Encode the data for it to be sent over the network with BytesIO
4. Decode the data once received back to a flat tensor
5. Assign that flat tensor to the model's weights with torch.nn.utils.vector_to_parameters(a*1e5, glob_net.parameters())
"""

def train_model(glob_net, opt, buffer, i, semaphor, res_queue):
    loc_net = ToyNet()

    b = ReplayBuffers(shared_replay_buffer=buffer,
                      cpu_id=i,
                      len_interaction=LEN_INPUTS_X + LEN_OUTPUTS_Y,
                      batch_size=BATCH_SIZE,  # If increased it's crap
                      num_iters=LEN_ITERATIONS,
                      tot_num_cpus=NUM_CPUS,
                      replacement=SAMPLE_WITH_REPLACEMENT,
                      sample_from_shared_memory=SAMPLE_FROM_SHARED_MEMORY)

    for i in range(NUM_EPISODES):
        # Generate training data and update buffer
        for j in range(NUM_STEPS):
            # Generates some data according to the data generative mechanism
            tensor_tuple = Manager.data_generative_mechanism()
            # Records the interaction inside the shared Tensor buffer
            b.record_interaction(tensor_tuple)

            # Every once in a while
            if (j + 1) % 2 == 0:
                # Waits for all of the cpus to provide a green light (min number of sampled item to begin process)
                if not NUM_EPISODES:
                    # Do this only for the first absolute run
                    Manager.wait_for_green_light(semaphor=semaphor, cpu_id=i)

                # Random samples a batch
                sampled_batch = b.random_sample_batch(from_shared_memory=True)
                # Forward pass of the neural net, until the output columns, in this case last one
                loc_output = loc_net.forward(sampled_batch[:, :-1])
                # Calculates the loss between target and predict
                loss = F.mse_loss(loc_output, torch.Tensor(sampled_batch[:, -1]).reshape(-1, 1))
                # Averages the loss if using batches, else only the single value
                res_queue.put(loss.mean().item())
                # Zeroes the gradients out
                opt.zero_grad()
                # Performs calculation of the backward pass
                loss.backward()

                # Perform the update of the global parameters using the local ones
                for lp, gp in zip(loc_net.parameters(), glob_net.parameters()):
                    gp._grad = lp.grad
                opt.step()

                loc_net.load_state_dict(glob_net.state_dict())
                print(f'EPISODE {i} STEP {j + 1} -> Loss for cpu {b.cpu_id} is: {loss}')

    res_queue.put(None)


if __name__ == '__main__':
    glob_net = ToyNet()
    glob_net.share_memory()

    opt = SharedAdam(glob_net.parameters(), lr=1e-3, betas=(0.92, 0.999))  # global optimizer

    # Queue used to store the history of rewards while training
    res_queue = mp.Queue()

    # Initializes the global buffer, where interaction with the environment are stored
    buffer = ReplayBuffers.init_global_buffer(len_interaction=LEN_INPUTS_X + LEN_OUTPUTS_Y,  # 2 inputs + 1 output
                                              num_iters=LEN_ITERATIONS,
                                              tot_num_cpus=NUM_CPUS,
                                              dtype=torch.float32)
    # Creates a starting semaphor
    semaphor = Manager.initialize_semaphor(NUM_CPUS)

    procs = [mp.Process(target=train_model, args=(glob_net, opt, buffer, i, semaphor, res_queue)) for i in
             range(NUM_CPUS)]

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
