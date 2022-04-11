import random

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from ReplayBuffer.buffer import ReplayBuffers
from Optims.shared_optims import SharedAdam
from main import Manager

mp.set_start_method('spawn', force=True)

LEN_SINGLE_STATE = 2
LEN_ITERATIONS = 100
NUM_CPUS = mp.cpu_count()

def f_true(x, y): return x ** 2 + y ** 2

# We want to approximate this function in the range [-5, 5]

def selected_range_sample(): return [(random.random() - .5) * 10, (random.random() - .5) * 10]

# Generate our data, where last column is output
def generate_data_row():
    d = selected_range_sample()
    return torch.Tensor([*d, f_true(*d)]).to(torch.float32)

class NNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


NUM_EPISODES = 500
NUM_STEPS = 50
def train_model(glob_net, opt, buffer, i, semaphor, res_queue):
    loc_net = NNet()

    b = ReplayBuffers(shared_replay_buffer=buffer,
                      cpu_id=i,
                      len_interaction=LEN_SINGLE_STATE + 1,
                      batch_size=10,
                      num_iters=LEN_ITERATIONS,
                      tot_num_cpus=NUM_CPUS,
                      replacement=False)

    semaphor[b.cpu_id] = True

    while not torch.all(semaphor):
        pass

    for i in range(NUM_EPISODES):
        # Generate training data and update buffer
        for j in range(NUM_STEPS):
            tensor_tuple = generate_data_row()
            b.record_interaction(tensor_tuple)
            batch = b.random_sample_batch(from_shared_memory=True)

            loc_output = loc_net.forward(batch[:, :-1])
            # Maybe remove this torch.Tensor, if one dimensional, indexing does not return tensor(?)
            loss = F.mse_loss(loc_output, batch[:, -1])
            res_queue.put(loss.mean().item())
            opt.zero_grad()
            loss.backward()
            if not((j+1) % 25):
                for lp, gp in zip(loc_net.parameters(), glob_net.parameters()):
                    gp._grad = lp.grad
                opt.step()

                loc_net.load_state_dict(glob_net.state_dict())
                print(f'EPISODE {i} STEP {j+1} -> Loss for cpu {b.cpu_id} is: {loss}')
    res_queue.put(None)

if __name__ == '__main__':
    glob_net = NNet()
    glob_net.share_memory()

    opt = SharedAdam(glob_net.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    res_queue = mp.Queue()

    buffer = ReplayBuffers.init_global_buffer(len_interaction=LEN_SINGLE_STATE + 1, # 2 inputs + 1 output
                                              num_iters=LEN_ITERATIONS,
                                              tot_num_cpus=NUM_CPUS,
                                              dtype=torch.float32)
    semaphor = Manager.initialize_semaphor(NUM_CPUS)

    procs = [mp.Process(target=train_model, args=(glob_net, opt, buffer, i, semaphor, res_queue)) for i in range(mp.cpu_count())]
    [p.start() for p in procs]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [p.join() for p in procs]

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()