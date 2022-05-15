import torch

class ActorCritic(object):

    @staticmethod
    def agent_step(env, neural_net, state, temporary_buffer):
        """Performs a step for the agent, choosing an action from the current state and acting"""
        # Transforms the numpy array state into a tensor object of float32
        state_tensor = torch.Tensor(state).to(torch.float32)

        # Choose an action using the network, using the current state as input
        action_chosen = neural_net.choose_action(state_tensor)

        # Prepares a list containing all the objects above
        tensor_tuple = [*state, action_chosen]

        # Note that this state is the next one observed, it will be used in the next iteration
        state, reward, done, _ = env.step(action_chosen)
        if done: reward = -1

        # Adds the reward experienced to the current episode reward
        # cum_reward += reward

        # Adds the reward and a placeholder for the discounted reward to be calculated
        tensor_tuple.append(reward)

        # Appends (state, action, reward, reward_observed) tensor object
        temporary_buffer.append(torch.Tensor(tensor_tuple))

        return state, reward, done

    @staticmethod
    def discount_rewards(neural_net, shared_memory_buffer, temporary_buffer, len_state, gamma, done):
        temporary_buffer.reverse()
        if done:
            R = 0
        else:
            _, output = neural_net.forward(temporary_buffer[-1][:len_state])
            # Output in this case is the estimation of the value coming from the state
            R = output.item()

        for idx, interaction in enumerate(temporary_buffer):
            # Take the true experienced reward from that session and the action taken in that step
            r = interaction[-1]
            a = interaction[-2]

            R = r + gamma * R
            shared_memory_buffer.record_interaction(torch.Tensor([*interaction[:len_state], a, R]).to(torch.float32))


