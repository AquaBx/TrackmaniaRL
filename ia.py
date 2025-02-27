import random
import numpy as np
import torch

from memory import Memory
from neural import Brain, Neural

def pickBestDevice():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

class IA:
    def __init__(self, input_dim : int, output_dim : int):

        self.total_reward = 0

        self.mutation_rate = 0.1

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.999975
        self.exploration_rate_min = 0.1

        self.curr_step = 0
        self.burnin = 1e5  # min. experiences before training
        self.learn_every = 3   # no. of experiences between updates to Q_online
        self.sync_every = 1e4   # no. of experiences between Q_target & Q_online sync

        self.memory = Memory(300,100000)

        self.device = pickBestDevice()
        self.brain = Brain(self.device,input_dim,output_dim)

        self.output_dim = output_dim

    def act(self, state : np.ndarray):
        if np.random.rand() < self.exploration_rate:
            action_idx = torch.randint(0,3,(len(self.output_dim),1))
        else :
            state = torch.tensor(state, device=self.device, dtype=torch.float)
            state = state.unsqueeze(0)
            actions_values = self.brain.act(state)
            action_idx = torch.argmax(actions_values,dim=2,keepdim=True)[0]


        # state = torch.tensor(state, device=self.device, dtype=torch.float)
        # state = state.unsqueeze(0)
        # actions_values = self.brain.act(state)
        # pb = torch.nn.functional.softmax(actions_values, dim=2)
        # action_idx = torch.multinomial(pb[0],1)

        # decrease exploration_rate
        # self.exploration_rate *= self.exploration_rate_decay
        # self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        
        return action_idx.detach().numpy()

    def mutate(self):
        for param in self.net.parameters():
            mutation_noise = torch.randn_like(param) * 0.5
            mask = torch.rand_like(param) < self.mutation_rate
            param.data += mask * mutation_noise

    def reward(self, score : float):
        self.total_reward += score

    def reset(self):
        self.total_reward = 0

    def save(self, save_dir):
        save_path = save_dir / f"net_best.pt"
        torch.save(
            dict(
                model=self.brain.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"Net saved to {save_path}")

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=self.device)
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.brain.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate

    def remember(self, state, next_state, action, reward, done):
        self.memory.store(state, next_state, action, reward, done)

    def short_term_learning(self,i):
        state, next_state, action, reward, done = self.memory.retrieve_last(i)
        return self.brain.learn(state, next_state, action, reward, done)

    def long_term_learning(self):
        size = len(self.memory.memory)
        if size < self.memory.batch_size:
            return 0

        state, next_state, action, reward, done = self.memory.retrieve()

        return self.brain.learn(state, next_state, action, reward, done)

