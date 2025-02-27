import dxcam
import cv2
import numpy as np
import numpy as np


class Viewer():
	def __init__(self, title):
		self.fps = 30
		self.title = title

		self.camera = dxcam.create(output_color="BGR")
		self.camera.start(target_fps=self.fps)

		self.frame = []
		self.projection_matrix = np.array([])
		self.windows_size = np.array([])
		self.windows_pos = np.array([])

		# cv2.namedWindow(title, cv2.WINDOW_NORMAL)

	def drawLine(self, start : np.array, vector : np.array):
		end = np.copy(start)

		for i in range(60):
			m = self.toScreen(end)
			x,y = int(m[0]),int(m[1])
			if self.frame[y][x] == 255:
				break
			end += vector

		return (start,end)

	def toScreen(self,pos):
		pos = np.concatenate((pos,[1]))
		ret = self.projection_matrix @ pos 
		
		if ret[3] == 0 :
			return np.array([0, 0, ret[3]])
		
		return self.window_pos + (ret[0:2] / ret[3] + 1) / 2 * self.window_size


	def updateSensors(self, points):
		self.sensors = [ self.drawLine(start, vector) for start,vector in points ]

	def updateFrame(self):
		image = self.camera.get_latest_frame()
		blackLow = np.array([0,0,0])
		blackHigh = np.array([90,90,90])

		self.frame = cv2.inRange(image,blackLow,blackHigh)

	def update(self, projection_matrix, window_size, window_pos, points):

		self.projection_matrix = projection_matrix
		self.window_pos = window_pos
		self.window_size = window_size

		# cv2.resizeWindow(self.title, window_size[0], window_size[1])
		self.updateFrame()
		self.updateSensors(points)

		return [np.linalg.norm(p2-p1) for p1,p2 in self.sensors]

	def render(self):
		img = self.frame
		for i,p in enumerate(self.sensors):
			p1 = [int(x) for x in self.toScreen(p[0])]
			p2 = [int(x) for x in self.toScreen(p[1])]

			cv2.putText(img, str(np.linalg.norm(p[1]-p[0])), (50, 50 + i * 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (125), 2, cv2.LINE_AA, False)

			cv2.line(img, p1,p2,(120), 5)

		cv2.imshow(self.title, img)
	
	def isOpened(self):
		if cv2.waitKey(1) & 0xFF == ord('q'):
			return False
		return True
	
	def destroy(self):
		cv2.destroyAllWindows()
		self.camera.stop()
