import os
import numpy as np
import torch

class Buffer:
    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n             = 0
        self._p             = 0
        self.total_size     = buffer_size

        self.device = device
        self.states         = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions        = torch.empty(
            (self.total_size, *action_shape), dtype=torch.int, device=device)
        self.next_states    = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.dones          = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)

    def append(self, state, action, next_state, done):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self.dones[self._p].copy_(torch.from_numpy(done))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.next_states[idxes],
            self.dones[idxes],
        )
    
    def load(self,path):
        tmp = torch.load(path)
        self._n = tmp['states'].size(0)
        self.total_size = self._n
        self.states             = tmp['states'].clone().to(self.device)
        self.actions            = tmp['actions'].clone().to(self.device)
        self.next_states        = tmp['next_states'].clone().to(self.device)
        self.dones              = tmp['dones'].clone().to(self.device)

    def save(self, path):
        os.makedirs(os.path.dirname(path),exist_ok=True)
        torch.save({
            'states': self.states.clone().cpu(),
            'actions': self.actions.clone().cpu(),
            'next_states': self.next_states.clone().cpu(),
            'dones': self.dones.clone().cpu(),
        }, path)
        
class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n             = 0
        self._p             = 0
        self.total_size     = buffer_size

        self.device = device
        self.states         = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions        = torch.empty(
            (self.total_size, *action_shape), dtype=torch.int, device=device)
        self.next_states    = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.rewards          = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones          = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis         = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)

    def append(self, state, action, next_state,reward, done,log_pi):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self.rewards[self._p].copy_(torch.from_numpy(reward))
        self.dones[self._p].copy_(torch.from_numpy(done))
        self.log_pis[self._p].copy_(torch.from_numpy(log_pi))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.next_states[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
        )

    def get(self):
        assert self._p % self.total_size == 0
        start = (self._p - self.total_size) % self.total_size
        idxes = slice(start, start + self.total_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.next_states[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
        )
