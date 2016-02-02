import antenna
import sys

def search(Positions, k, c):
	poles = list(Positions)

	# Add an antenna on the first pole
	antennas = []
	antennas.append(antenna.Antenna(poles[0], k, c))
	poles.remove(poles[0])
	antennaIdx = 0

	# Iterate as long as the problem isn't solved
	solutionFound = 0
	while(solutionFound == 0):
		distanceMin = sys.maxint
		closestPosition = -1
		for i in range(0, len(poles)):
			distanceTmp = antennas[antennaIdx].distanceTo(poles[i])
			if(distanceTmp < distanceMin):
				distanceMin	= distanceTmp
				closestPosition = i

		if(closestPosition == -1):
			solutionFound = 1
		else:
			if(distanceMin <= (2*antennas[antennaIdx].rMax)):
				#Adding another pole to this antenna range
				antennas[antennaIdx].addPole(poles[closestPosition])
			else:
				#Adding a new antenna since expanding the previous one isn't worth it
				antennas.append(antenna.Antenna(poles[closestPosition], k, c))
				antennaIdx += 1

			#Removing the current pole from the list, since it's been served
			poles.remove(poles[closestPosition])

	#Once we're done, we check to see if any optimisation can be done about the minimum length of each 'r'
	cost = 0
	for i in range(0, len(antennas)):
		antennas[i].shrinkRange()
		antennas[i].printData()
		cost += antennas[i].calculateCost()
	print('Cost: ' + str(cost))

def main():
	#search([(10,10),(20,20),(30,0),(30,40),(50,40)],200,1)
	search([(23,4),(43,43),(54,54),(54,94),(24,54),(54,52),(34,23),(76,76),(87,98),(98,9),(56,6),(53,4),(23,3),(45,3),(65,4),(7,8)], 200, 1)


if __name__ == "__main__":
	main()
