import antenna
import sys

def search(Positions, k, c):

	# Add an antenna on every pole
	antennas = []
	for i in Positions:
		antennas.append(antenna.Antenna(i, k, c))

	# Iterate as long as the problem isn't solved
	solutionFound = 0
	while(solutionFound == 0):
		# Check if two antennas overlap
		# if they do and are the closest, save them
		overlapping = []
		# Since we start all antennas with a range of rMax = ceil(sqrt(K/C)) we can assume minOverlapDist must be 2*rMax
		minOverlapDist = 2*antennas[0].rMax
		for i in range(0, len(antennas)):
			antennaI = antennas[i]
			for j in range(i+1, len(antennas)):
				antennaJ = antennas[j]
				distanceIJ = antennaI.distanceTo(antennaJ.position)
				if distanceIJ <= minOverlapDist and antennaI.canMergeAntenna(antennaJ):
					overlapping = [i, j]
					minOverlapDist = distanceIJ

		# Merge the two closest antennas, if they exist
		if overlapping:
			antennaJ = antennas[overlapping[1]]
			antennas.remove(antennaJ)
			antennas[overlapping[0]].mergeAntenna(antennaJ)
		else:
			solutionFound = 1

	#Once we're done, we check to see if any optimisation can be done about the minimum length of each 'r'
	cost = 0
	antenaList = []
	for i in range(0, len(antennas)):
		antennas[i].shrinkRange()
		#antennas[i].printData()
		cost += antennas[i].calculateCost()
		antenaList.append((antennas[i].position[0], antennas[i].position[1], int(antennas[i].rMax)))
	print('Cost: ' + str(cost))
	return antenaList

def main():
	search([(10,10),(20,20),(30,0),(30,40),(50,40)],200,1)


if __name__ == "__main__":
	main()
