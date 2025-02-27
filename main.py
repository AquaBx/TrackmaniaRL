from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from env import TMEnv
from ia import IA
from viewer import Viewer
from population import Population

population = Population(IA, 1, 100, 15 , [3,3,3])

# population.individus[0].load(Path("checkpoints/net_best.pt"))

posSave = False

env = TMEnv(posSave)

if posSave:
	while env.viewer.isOpened():
		env.update()
		env.render()
		# env.savePosition()
else:
	while not( population.finished() ):
		state, info = env.reset()
		actual = population.get_actual()
		cooldown = 0

		while env.viewer.isOpened():
			while not(env.raceStarted):
				env.update()

			action = actual.act(state)
		
			next_state, reward, terminated, truncated, info = env.step(action)

			# actual.reward(reward)

			actual.remember(state, next_state, action, reward, terminated)

			loss = actual.long_term_learning()
			print(loss,actual.exploration_rate)

			if terminated or truncated:
				break

			state = next_state

			# env.render()
			cooldown += 1
			
		actual.exploration_rate = np.random.rand()
		population.step()

env.close()

save_dir = Path('checkpoints')
if not( save_dir.exists() ):
    save_dir.mkdir(parents=True)

population.individus[0].save(save_dir)