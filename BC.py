import torch
from torch import nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from torch.optim import Adam

from torch.distributions.categorical import Categorical
from env import *
from buffer import Buffer


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

def main():
    device = 'mps'
    env = gym.make('GridWorld-v0', grid_size=16,seed=43)
    state_shape = env.observation_space.shape
    action_shape = (1,)
    batch_size = 128
    
    expert_buffer = Buffer(buffer_size=1,state_shape=state_shape,
                           action_shape=action_shape,device=device)
    expert_buffer.load('./dataset/dataset_10000.pth')
    policy = StateIndependentPolicy_discrete_2D(state_shape=state_shape,
                        action_shape=action_shape,hidden_units=[64,64],
                        hidden_activation=nn.ReLU()).to(device)
    
    optimizer = Adam(policy.parameters(), lr=1e-4)
    
    for iter in range(int(1e6)):
        (states,actions,_,_) = expert_buffer.sample(batch_size=batch_size)
        probs = policy.evaluate_prob(states, actions)    
        loss = -probs.mean()
        
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if (iter%1000 == 0):
            print(iter,probs.mean().item())
            returns = []
            policy.eval()
            with torch.no_grad():
                for ep in range(20):
                    state,_ = env.reset()
                    done = False
                    while(not done):
                        input = torch.tensor(state).to(device).unsqueeze(0)
                        action = policy(input).cpu().numpy()[0]
                        next_state, reward, done,_,_ = env.step(action)
                        if (done):
                            break
                        state = next_state
                    returns.append(reward)
            print(f'[Eval]: {np.mean(returns):.2f}')
            policy.train()
            
    pass

if __name__ == '__main__':
    main()