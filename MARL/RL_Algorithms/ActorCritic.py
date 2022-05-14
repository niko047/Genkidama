import torch

class ActorCritic(object):

    def __init__(self, env, neural_net):
        self.env = env
        self.neural_net = neural_net


    def agent_step(self, state, temporary_buffer):
        """Performs a step for the agent, choosing an action from the current state and acting"""
        # Transforms the numpy array state into a tensor object of float32
        state_tensor = torch.Tensor(state).to(torch.float32)

        # Choose an action using the network, using the current state as input
        action_chosen = self.neural_net.choose_action(state_tensor)

        # Prepares a list containing all the objects above
        tensor_tuple = [*state, action_chosen]

        # Note that this state is the next one observed, it will be used in the next iteration
        state, reward, done, _ = self.env.step(action_chosen)
        if done: reward = -1

        # Adds the reward experienced to the current episode reward
        # cum_reward += reward

        # Adds the reward and a placeholder for the discounted reward to be calculated
        tensor_tuple.append(reward)

        # Appends (state, action, reward, reward_observed) tensor object
        temporary_buffer.append(torch.Tensor(tensor_tuple))

        return done

    def empty_temporary_buffer(self):
        self.temporary_buffer = []

