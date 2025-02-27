import torch
import numpy as np

from ia import IA

class Population:
    def __init__(self, ia : type[IA], individus_count : int, generations_count : int, input_size : int, output_size : int):
        self.ia_constructor = ia

        self.input_size = input_size
        self.output_size = output_size

        self.individus = [ self.ia_constructor(input_size,output_size) for _ in range(individus_count)]

        self.individus_actual = 0
        self.generations_actual = 0

        self.individus_count = individus_count
        self.generations_count = generations_count

    def crossover(self, parent1 : IA, parent2 : IA):
        child1 = self.ia_constructor(self.input_size,self.output_size)
        child2 = self.ia_constructor(self.input_size,self.output_size)
        
        parent1_params = [param.data for param in parent1.net.parameters()]
        parent2_params = [param.data for param in parent2.net.parameters()]

        for child in [child1, child2]:
            for n, param in enumerate(child.net.parameters()):
                tp1 = parent1_params[n]
                tp2 = parent2_params[n]

                # Create a mask with 0.5 probability
                mask = torch.rand_like(tp1) < 0.5

                # Apply crossover using the mask
                param.data.copy_(torch.where(mask, tp1, tp2))

        return child1, child2

    def get_bests(self,n : int):
        return self.individus[0:n]

    def get_randoms(self,n : int):
        score_v = np.array([x.total_reward for x in self.individus])
        score_v -= score_v[-1]
        score_v = score_v / np.sum(score_v)

        return np.random.choice(self.individus, n, p=score_v)

    def next_generation(self):

        if self.individus_count == 1:
            return

        bests_count = (self.individus_count >> 3)  # 12.5%
        bests_count = bests_count >> 1 << 1 # tric pour être pair
        next_individus = self.get_bests(bests_count)

        assert len(next_individus)%2 == 0

        randoms = self.get_randoms(self.individus_count-bests_count) # reste en random pondéré

        for i in range(0,len(randoms),2):
            child1,child2 = self.crossover(randoms[i],randoms[i+1])
            child1.mutate()
            child2.mutate()
            next_individus.append(child1)
            next_individus.append(child2)

        assert len(next_individus) == self.individus_count

        for x in next_individus:
            x.reset()

        self.individus = next_individus

    def get_actual(self):
        return self.individus[self.individus_actual]

    def step(self):
        if self.individus_actual == self.individus_count - 1:
            self.individus_actual = 0
            self.generations_actual += 1
            self.individus.sort(reverse=True,key=lambda ia:ia.total_reward)
            # print([x.total_reward for x in self.individus])
            self.next_generation()
        else:
            self.individus_actual += 1

    def finished(self):
        return self.generations_actual == self.generations_count