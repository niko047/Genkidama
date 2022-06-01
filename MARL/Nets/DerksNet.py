import torch.nn as nn
from .general_network import GeneralNeuralNet
import torch
import torch.nn.functional as F

class DerksNet(nn.Module, GeneralNeuralNet):
    def __init__(self, s_dim, a_dim):
        super(DerksNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, 128)
        self.pi3 = nn.Linear(128, a_dim)

        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 128)
        self.v3 = nn.Linear(128, 1)
        DerksNet.set_init([self.pi1, self.pi2, self.pi3, self.v1, self.v2, self.v3])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):

        pi1_ = F.relu(self.pi1(x))
        pi2_ = F.relu(self.pi2(pi1_))
        logits = self.pi3(pi2_)
        v1_ = F.relu(self.v1(x))
        v2_ = F.relu(self.v2(v1_))
        values = self.v3(v2_)

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
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

    @staticmethod
    def set_init(layers):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0., std=1)
            nn.init.constant_(layer.bias, 0.)

