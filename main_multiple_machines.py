from MARL.CoresOrchestrator import CoresOrchestrator
from MARL.Nets.neural_net import ToyNet
from MARL.Optims.shared_optims import SharedAdam


# TODO - Set up a different environment for these ones
neural_net = ToyNet
shared_optimizer = SharedAdam
shared_opt_kwargs = {
    "lr": 1e-3,
    "betas": (0.92, 0.999)
}
len_interaction_X = 2
len_interaction_Y = 1
batch_size = 3
num_iters = 50
replacement = False
sample_from_shared_memory = True
cpu_capacity = .80  # 80%
num_steps = 200
num_episodes = 100

# Alpha is the parameter determining the importance of the individual cores when sending weights to parent net
# TODO - Insert alpha inside the function
alpha = 1

if __name__ == '__main__':


    machine = CoresOrchestrator(neural_net=neural_net,
                                shared_optimizer=shared_optimizer,
                                shared_optimizer_kwargs=shared_opt_kwargs,
                                len_interaction_X=len_interaction_X,
                                len_interaction_Y=len_interaction_Y,
                                batch_size=batch_size,
                                num_iters=num_iters,
                                replacement=replacement,
                                sample_from_shared_memory=sample_from_shared_memory,
                                cpu_capacity=cpu_capacity,
                                num_steps=num_steps,
                                num_episodes=num_episodes)
    machine.run_procs()