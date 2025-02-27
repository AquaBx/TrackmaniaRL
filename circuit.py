import numpy as np

class Circuit:
    def __init__(self):
        self.circuit = self._loadCircuit()
        self.circuitIndex = 0
        self.plan = np.array([0,0,0,0])
        self.dir = np.array([0,0,0])
        self.nextCheckpoint()

    def nextCheckpoint(self):
        p1 = self.circuit[self.circuitIndex]
        p2 = self.circuit[self.circuitIndex+1]

        self.dir = (p2-p1).transpose()
        d = - p1.dot(self.dir)
        self.plan = np.append(self.dir,d)

        self.circuitIndex += 1


    def distanceCheckpoint(self, p):
        p2 = np.append(p,1)
        dir2 = np.append(self.dir,0)
        
        return np.dot(self.plan,p2)/np.dot(self.plan,dir2)*np.linalg.norm(self.dir)

    def distanceToNextCheckpoint(self, p):
        distance = self.distanceCheckpoint(p)
        while distance >= 0:
            self.nextCheckpoint()
            distance = self.distanceCheckpoint(p)
        return distance
    
    def distanceBetweenCheckpoints(self, i1, i2):
        distance = 0
        for i in range(i1,i2):
            distance += np.linalg.norm(self.circuit[i+1]-self.circuit[i])
        return distance

    def isFinished(self):
        return self.circuitIndex >= len(self.circuit)-2

    def reset(self):
        self.circuitIndex = 0
        self.nextCheckpoint()

    def _loadCircuit(self):
        with open('position.txt','r') as f:
            return [ np.array([float(c) for c in l.split(",")]) for l in f.read().split("\n") if l != "" ]


# def direction(p1,p2):
#     return p2-p1

# def plan(p1 : np.array,p2 : np.array):
#     dir = direction(p1,p2).transpose()
#     d = - p1.dot(dir)
#     coefs = np.append(dir,d)
#     return coefs,dir



# p,d = plan(np.array([1,1,1]),np.array([2,2,2]))

# isBehind(p,d,np.array([1,0.9,1]))

