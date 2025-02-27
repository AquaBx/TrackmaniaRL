import socket
import struct
from threading import Thread
import time

import numpy as np
import gymnasium
import GameControl
from circuit import Circuit
from viewer import Viewer

class TMEnv(gymnasium.Env):
	def __init__(self, posSave : bool ):
		self.viewer = Viewer("Window")
		
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.connect(('localhost',12345))

		self.speed = 0
		self.x = 0
		self.y = 0
		self.z = 0
		self.running = True
		self.raceStarted = False
		self.sensors = []

		self.cooldown = 0
		self.lastProgression = 0

		if not(posSave):
			self.gamepad = GameControl.createGamepad()
			self.circuit = Circuit()

	def update(self):
		self.socket.send("r".encode())

		rec  = self.socket.recv(16)
		screen_w, screen_h, screen_x, screen_y = struct.unpack('iiii', rec[0:16])

		rec  = self.socket.recv(16)
		self.x, self.y, self.z, left_x = struct.unpack('ffff', rec[0:16])

		rec  = self.socket.recv(16)
		left_y, left_z, top_x, top_y = struct.unpack('ffff', rec[0:16])

		rec  = self.socket.recv(16)
		top_z, dir_x, dir_y, dir_z = struct.unpack('ffff', rec[0:16])

		rec  = self.socket.recv(16)
		proj_1 = struct.unpack('ffff', rec[0:16])

		rec  = self.socket.recv(16)
		proj_2 = struct.unpack('ffff', rec[0:16])

		rec  = self.socket.recv(16)
		proj_3 = struct.unpack('ffff', rec[0:16])

		rec  = self.socket.recv(16)
		proj_4 = struct.unpack('ffff', rec[0:16])

		projection_matrix = np.array([proj_1,proj_2,proj_3,proj_4])
		display_size = np.array([screen_w,screen_h])
		display_pos = np.array([screen_x,screen_y])

		position = np.array([self.x,self.y,self.z])
		left = np.array([left_x,left_y,left_z])
		top = np.array([top_x,top_y,top_z])
		dir = np.array([dir_x,dir_y,dir_z])

		rec  = self.socket.recv(5)
		self.raceStarted,self.speed = struct.unpack('fb', rec)

		devant = position + dir*2.1
		d_devant = dir*4
		d_left = left*1.1
	
		points = []

		for i in range(-1,6):
			p1 = devant - d_left
			v1 = d_devant - d_left * (i - 1)
			v1 /= np.linalg.norm(v1)
			
			p2 = devant + d_left
			v2 = d_devant + d_left * (i - 1)
			v2 /= np.linalg.norm(v2)

			points.append((p1,v1))
			points.append((p2,v2))

		self.sensors = self.viewer.update(projection_matrix,display_size,display_pos, points)

		return [self.speed/1000] + self.sensors

	def step(self, action: list[int]):
		a,b,c = action
		a,b,c = a[0],b[0],c[0]
		GameControl.play(self.gamepad,a,b,c)

		t1 = time.time()

		state = self.update()

		t2 = time.time()

		dt = (t2-t1)

		distanceToNext = self.circuit.distanceToNextCheckpoint(np.array([self.x,self.y,self.z]))
		distanceNext = self.circuit.distanceBetweenCheckpoints(0,self.circuit.circuitIndex)
		actualProgression = distanceToNext + distanceNext

		score = (actualProgression - self.lastProgression) / dt

		self.lastProgression = actualProgression

		# thresh = min(sensors[1], sensors[-2])

		# if thresh < 0.6:
		# 	score = - 10 * (1 - thresh)

		if score <= 0 :
			self.cooldown += 1
	

		done = self.circuit.isFinished()
		truncated = self.cooldown > 10

		score = 0 if any([x == 0 for x in self.sensors]) else score

		return state, float(score), done , truncated , {}

	def reset(self):
		GameControl.reset(self.gamepad)
		self.circuit.reset()

		self.cooldown = 0
		self.lastProgression = 0

		state = self.update()

		return state, {}

	def close(self):
		self.runing = False
		self.socket.close()
		self.viewer.destroy()

	def savePosition(self):
		with open('position.txt','a+') as f:
			f.write(f'{self.x},{self.y},{self.z}\n')

	def render(self):
		self.viewer.render()

		