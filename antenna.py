import math
import sys

class Antenna:
	def __init__(self, position, costPerAntenna, costPerDistance):
		self.position = position
		# Start rMax at the maximal "worth" distance and shrink later
		self.rMax = 1 + math.sqrt(costPerAntenna / costPerDistance)
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

	def distanceTo(self, position):
		return math.sqrt(pow(self.position[0] - position[0], 2) + pow(self.position[1] - position[1], 2))

	def calculateLinkCost(self, polePosition, c):
		return c * distance(polePosition)

	def mergeAntenna(self, antenna):
		self.poles.extend(antenna.poles)
		self.recalculatePosition()

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
			self.rMax = tempRange

	def printData(self):
		print(str('Position: ' + str(self.position) + ', range: ' + str(self.rMax)))
