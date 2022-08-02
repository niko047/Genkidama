import torch.nn
import torch.nn as nn
import torch.nn.functional as F

from .general_network import GeneralNeuralNet

class SmallNet(nn.Module, GeneralNeuralNet):
    def __init__(self, s_dim, a_dim):
        super(SmallNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        SmallNet.set_init([self.pi1, self.pi2, self.v1, self.v2])
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
        #if self.epsilon_greedy:

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