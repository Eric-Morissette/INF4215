import math
import sys

def distanceBetween(pos1, pos2):
	return math.sqrt(pow(pos1[0] - pos2[0], 2) + pow(pos1[1] - pos2[1], 2))

class Antenna:
	def __init__(self, position, costPerAntenna, costPerDistance):
		self.position = position
		# Start rMax at the maximal "worth" distance and shrink later
		self.rMax = math.ceil(math.sqrt(costPerAntenna / costPerDistance))
		self.k = costPerAntenna
		self.c = costPerDistance
		self.poles = []
		self.poles.append(position)

	def recalculatePosition(self):
		xMax = -sys.maxint
		yMax = -sys.maxint
		xMin = sys.maxint
		yMin = sys.maxint
		size = len(self.poles)
		for i in self.poles:
			xMax = max(xMax, i[0])
			xMin = min(xMin, i[0])
			yMax = max(yMax, i[1])
			yMin = min(yMin, i[1])
		self.position = (xMin + ((xMax-xMin)/2), yMin + ((yMax-yMin)/2))

	def addPole(self, polePosition):
		self.poles.append(polePosition)
		self.recalculatePosition()

	def removePole(self, polePosition):
		self.poles.remove(polePosition)
		self.recalculatePosition()

	def simulateRadius(self, positions):
		xMax = -sys.maxint
		yMax = -sys.maxint
		xMin = sys.maxint
		yMin = sys.maxint
		for i in positions:
			xMax = max(xMax, i[0])
			xMin = min(xMin, i[0])
			yMax = max(yMax, i[1])
			yMin = min(yMin, i[1])
		positionTemp = (xMin + ((xMax-xMin)/2), yMin + ((yMax-yMin)/2))

		maxDist = -1
		for i in positions:
			maxDist = max(maxDist, distanceBetween(positionTemp, i))
		return maxDist

	def canAddPole(self, polePosition):
		polesList = list(self.poles)
		polesList.append(polePosition)
		return (self.simulateRadius(polesList) < self.rMax)

	def distanceTo(self, position):
		return math.sqrt(pow(self.position[0] - position[0], 2) + pow(self.position[1] - position[1], 2))

	def mergeAntenna(self, antenna):
		self.poles.extend(antenna.poles)
		self.recalculatePosition()

	def canMergeAntenna(self, antenna2):
		polesList = list(self.poles)
		polesList.extend(list(antenna2.poles))
		return (self.simulateRadius(polesList) <= self.rMax)

	def isOverlapping(self, antenna):
		# Si la somme des rayons est plus grande que la distance, 
		# alors on a un triangle avec dist - r1 - r2 et on overlap
		return (self.rMax + antenna.rMax) >= self.distanceTo(antenna.position)

	#Fonction utile en fin de recherche afin de minimiser la taille des antennes tout en restant une solution valide
	#utile puisqu'on place les antennes avec un rayon rMax, qui represente la distance optimale en terme de cout
	def shrinkRange(self):
		tempRange = 1
		for i in range(0, len(self.poles)):
			tempRange = max(tempRange, self.distanceTo(self.poles[i]))
		if tempRange < self.rMax:
			self.rMax = math.ceil(tempRange)

	def printData(self):
		print(str('Position: ' + str(self.position) + ', range: ' + str(self.rMax)))

	def calculateCost(self):
		return(self.k + self.c*(self.rMax * self.rMax))
