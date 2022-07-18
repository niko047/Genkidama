import threading
import torch.nn
import torch.nn as nn
import torch.nn.functional as F

from MARL.Nets.general_network import GeneralNeuralNet

class TestNet(nn.Module, GeneralNeuralNet):
    def __init__(self):
        super(TestNet, self).__init__()
        self.s_dim = 1
        self.a_dim = 1
        self.pi1 = nn.Linear(self.s_dim, 1)
        self.pi2 = nn.Linear(1, 1)
        self.v1 = nn.Linear(self.a_dim, 1)
        self.v2 = nn.Linear(1, 1)
        TestNet.set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=0).data
        m = self.distribution(prob)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        # Takes actual values of states (discounted) and does MSE against the estimated ones by the network
        td = v_t.reshape(-1, 1) - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss.ravel() + a_loss).mean()
        return total_loss


    @staticmethod
    def set_init(layers):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)

class Counter:
    def __init__(self):
        self.gcounter = 0

    def add1(self):
        for i in range(100):
            self.gcounter += 1


if __name__ == '__main__':
    c = TestNet(1,1)
    threads = [threading.Thread(target=c.add1) for j in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    print(f'Count is {c.gcounter}')