import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from buffer import ReplayBuffers
from main import Manager
import torch.multiprocessing as mp


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
    return torch.Tensor(d + [f_true(*d)], dtype=torch.float32)

class NNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=8)
        self.fc3 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


NUM_EPISODES = 500
NUM_STEPS = 50
def train_model(net, buffer, i, semaphor, queue):
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
        for j in range(NUM_EPISODES):
            tensor_tuple = generate_data_row()
            buffer.record_interaction(tensor_tuple)
        pass


if __name__ == '__main__':
    net = NNet()
    net.share_memory()

    buffer = ReplayBuffers.init_global_buffer(len_interaction=LEN_SINGLE_STATE + 1, # 2 inputs + 1 output
                                              num_iters=LEN_ITERATIONS,
                                              tot_num_cpus=NUM_CPUS,
                                              dtype=torch.float32)
    semaphor = Manager.initialize_semaphor(NUM_CPUS)
    queue = Manager.initialize_queue(NUM_CPUS)

    procs = [mp.Process(target=train_model, args=(net, buffer, i, semaphor, queue)) for i in range(mp.cpu_count())]
    [p.start() for p in procs]
    [p.join() for p in procs]


    optimizer = optim.Adam(n.parameters(), lr=1e-3)
    losses = []
    for k, (vx, vy) in enumerate(zip(X, y)):
        n.zero_grad()
        output = n(torch.Tensor(vx))
        loss = F.mse_loss(output, torch.Tensor([vy]))
        loss.backward()
        optimizer.step()
        losses.append(loss)
        print(f'Current loss is {loss}')
