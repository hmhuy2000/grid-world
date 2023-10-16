import torch
from torch import nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from torch.optim import Adam

from torch.distributions.categorical import Categorical
from env import *
from buffer import Buffer,RolloutBuffer


def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)

class StateIndependentPolicy_discrete_2D(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.ReLU()):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=3,padding='same'),
            hidden_activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3,padding='same'),
            hidden_activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3,padding='same'),
            hidden_activation,
            )

        self.fc_layers = build_mlp(
            input_dim=64 * state_shape[1] * state_shape[2]//16,
            output_dim=4,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def forward(self, states):
        x = self.conv_layers(states)
        x = x.view(x.size(0), -1)  # Flatten the states
        probs = F.softmax(self.fc_layers(x),dim=-1)
        return torch.argmax(probs,dim=-1)

    def sample(self, states):
        x = self.conv_layers(states)
        x = x.view(x.size(0), -1)  # Flatten the states
        probs = F.softmax(self.fc_layers(x),dim=-1)
        dis_ = Categorical(probs=probs)
        actions = dis_.sample()
        return actions, dis_.log_prob(actions)

    def evaluate_prob(self, states, actions):
        x = self.conv_layers(states)
        x = x.view(x.size(0), -1)  # Flatten the states
        probs = F.softmax(self.fc_layers(x),dim=-1)
        actions = actions.to(torch.int64)
        output = torch.gather(probs, 1, actions)
        return output
    
    def evaluate_log_pi(self, states, actions):
        x = self.conv_layers(states)
        x = x.view(x.size(0), -1)  # Flatten the states
        probs = F.softmax(self.fc_layers(x),dim=-1)
        actions = actions.to(torch.int64)
        output = torch.gather(probs, 1, actions)
        return torch.log(output)

class StateFunction_2D(nn.Module):

    def __init__(self, state_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh(),output_activation=None,add_dim=0):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=3,padding='same'),
            hidden_activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3,padding='same'),
            hidden_activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3,padding='same'),
            hidden_activation,
            )

        self.fc_layers = build_mlp(
            input_dim=64 * state_shape[1] * state_shape[2]//16,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def forward(self, states):
        x = self.conv_layers(states)
        x = x.view(x.size(0), -1)  # Flatten the states
        x = self.fc_layers(x)
        return x

class GAIL_disc_2D(nn.Module):

    def __init__(self, state_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh(),output_activation=None,add_dim=0):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=3,padding='same'),
            hidden_activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3,padding='same'),
            hidden_activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3,padding='same'),
            hidden_activation,
            )

        self.fc_layers = build_mlp(
            input_dim=64 * state_shape[1] * state_shape[2]//16,
            output_dim=4,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def forward(self, states,actions):
        x = self.conv_layers(states)
        x = x.view(x.size(0), -1)  # Flatten the states
        x = self.fc_layers(x)
        actions = actions.to(torch.int64)
        output = torch.gather(x, 1, actions)
        return output
    
    def calculate_reward(self, states, actions):
        with torch.no_grad():
            return - F.logsigmoid(-self.forward(states, actions)) 
    
def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)

def main():
    device = 'cpu'
    env = gym.make('GridWorld-v0', grid_size=16,seed=43)
    state_shape = env.observation_space.shape
    action_shape = (1,)
    batch_size = 128
    buffer_size = 10000
    epoch_disc = 40
    epoch_ppo = 40
    gamma = 0.95
    lambd = 0.97
    
    expert_buffer = Buffer(buffer_size=1,state_shape=state_shape,
                        action_shape=action_shape,device=device)
    expert_buffer.load('./dataset/dataset_10000.pth')
    roll_buffer = RolloutBuffer(buffer_size=buffer_size,state_shape=state_shape,
                        action_shape=action_shape,device=device,)
    
    policy = StateIndependentPolicy_discrete_2D(state_shape=state_shape,
                        action_shape=action_shape,hidden_units=[64,64],
                        hidden_activation=nn.ReLU()).to(device)
    critic = StateFunction_2D(state_shape=state_shape,
                        hidden_units=[64,64],hidden_activation=nn.ReLU(),
                        ).to(device)
    disc = GAIL_disc_2D(state_shape=state_shape,hidden_units=[64,64],
                        hidden_activation=nn.ReLU()).to(device)
    
    policy_optimizer = Adam(policy.parameters(), lr=1e-4)
    critic_optimizer = Adam(critic.parameters(), lr=1e-4)
    disc_optimizer = Adam(disc.parameters(), lr=1e-3)
    from tqdm import trange
    for train_step in range(10000):
        returns = [0.0]
        state,_ = env.reset()
        done = False
        for rollout_step in range(buffer_size):
            input = torch.tensor(state).to(device).unsqueeze(0)
            action,log_pi = policy.sample(input)
            action = action.detach().cpu().numpy()[0]
            log_pi = log_pi.detach().cpu().numpy()[0]
            next_state, reward, done,_,_ = env.step(action)
            returns[-1] += reward
            roll_buffer.append(state=state,action=np.array([action]),
                               next_state=next_state,reward=np.array([reward]),done=np.array([done]),
                               log_pi=np.array([log_pi]))
            if (done):
                next_state,_ = env.reset()
                done = False
                returns.append(0.0)
            state = next_state
            
        for _ in trange(epoch_disc):
            pi_states, pi_actions = roll_buffer.sample(batch_size)[:2]
            exp_states, exp_actions = expert_buffer.sample(batch_size)[:2]
            exp_log_pi = policy.evaluate_log_pi(exp_states, exp_actions).mean().item()
            logits_pi = disc(pi_states, pi_actions)
            exp_logits = disc(exp_states, exp_actions)
            loss_pi = -F.logsigmoid(-logits_pi).mean()
            exp_loss = -F.logsigmoid(exp_logits).mean()
            loss_disc = loss_pi + exp_loss
            disc_optimizer.zero_grad()
            loss_disc.backward()
            disc_optimizer.step()
        
        _states,_actions,_next_states,_rewards,_dones,old_log_pis = roll_buffer.get()
        # _rewards = disc.calculate_reward(_states,_actions)
        
        with torch.no_grad():
            values = critic(_states)
            next_values = critic(_next_states)
        targets, gaes = calculate_gae(
            values, _rewards, _dones, next_values, gamma, lambd)
        
        for _ in trange(epoch_ppo):
            loss_critic = (critic(_states) - targets).pow_(2).mean()
            critic_optimizer.zero_grad()
            loss_critic.backward()
            critic_optimizer.step()
            
        for _ in trange(epoch_ppo):
            log_pis = policy.evaluate_log_pi(_states, _actions)
            ratios = (log_pis - old_log_pis).exp_()
            entropy = -log_pis.mean()
            
            loss_actor1 = -ratios * gaes
            loss_actor2 = -torch.clamp(
                ratios,
                1.0 - 0.2,
                1.0 + 0.2
            ) * gaes
            policy_optimizer.zero_grad()
            loss_actor = torch.max(loss_actor1, loss_actor2).mean() - 0.0001 * entropy
            loss_actor.backward()
            policy_optimizer.step()
        print()
        print(f'{loss_disc.item():.2f},{loss_critic.item():.2f},{loss_actor.item():.2f}')
        print(f'{np.mean(returns):.2f},{_rewards.mean():.2f}')
        print(f'{old_log_pis.mean().item()},{exp_log_pi:.2f}')
        
    
    
if __name__ == '__main__':
    main()