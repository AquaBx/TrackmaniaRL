import copy
import numpy as np
from torch import nn
import torch

class Neural(nn.Module):
    # https://arxiv.org/pdf/1711.08946v2

    def __init__(self, input_dim : int, outputs_dim : list[int]):
        super().__init__()

        self.shared_representation = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.state_value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.branches = [ 
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output),
            ) 
            for output in outputs_dim
        ]

    def forward(self, input_data : torch.Tensor):
        shared = self.shared_representation(input_data)

        Advantages = torch.stack([model(shared) for model in self.branches])
        Mean = torch.mean(Advantages, dim=0)
        State = self.state_value(shared)

        return (Advantages-Mean).add(State).rot90(1, [0, 1])

class DoubleNeural(nn.Module):
    def __init__(self, input_dim : int, outputs_dim : list[int]):
        super().__init__()

        self.online = Neural(input_dim,outputs_dim)
        self.target = copy.deepcopy(self.online)
        self.tau = 0.1

        for p in self.target.parameters():
            p.requires_grad = False

    def align_target_network(self):
        for target_param, primary_param in zip( self.target.parameters(), self.online.parameters()):
            target_param.data.copy_(self.tau * primary_param.data + (1.0 - self.tau) * target_param.data)

    def forward(self, input, model):
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)

class Brain():
    def __init__(self,device,input_dim,output_dim):
        self.gamma = 0.99
        self.net = DoubleNeural(input_dim,output_dim).float().to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.criterion = torch.nn.MSELoss()
        self.device = device

    def act(self,state):
        return self.net(state, model="online")

    def learn(self, state, next_state, action, reward, done):
        # https://arxiv.org/pdf/1711.08946v2
        size = len(done)

        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device)
        
        Prev = self.act(state)
        # print(Prev)
        # print(action)
        Qd = torch.gather(Prev,2,action)
        # print(Qd)

        Next = self.act(next_state)
        actionNext = torch.argmax(Next,dim=2,keepdim=True)

        with torch.no_grad():
            # print("Qd m")
            Qd_m = self.net(next_state, model='target')
            # print(Qd_m)
            # print(best)
            Qd_m = torch.gather(Qd_m,2,actionNext)
            # print(Qd_m)
            # avg = torch.mean(Qd_m, dim=1)
            # print(avg)
            y = reward.view((size,1,1)) + self.gamma * Qd_m
            # print(y)

        # print(Qd.size(),y.size())


        self.optimizer.zero_grad()
        loss = self.criterion(Qd, y)
        loss.backward()

        self.optimizer.step()

        # print(self.net.parameters())

        self.net.align_target_network()

        # print(self.net.parameters())

        return loss.item()