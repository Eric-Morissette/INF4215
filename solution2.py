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
		minOverlapDist = sys.maxint
		for i in range(0, len(antennas)):
			antennaI = antennas[i]
			for j in range(i+1, len(antennas)):
				antennaJ = antennas[j]
				if antennaI.isOverlapping(antennaJ) and antennaI.distanceTo(antennaJ.position) <= minOverlapDist:
					overlapping = [i, j]
					minOverlapDist = antennaI.distanceTo(antennaJ.position)

		# Merge the two closest antennas, if they exist
		if overlapping:
			antennaJ = antennas[overlapping[1]]
			antennas.remove(antennaJ)
			antennas[overlapping[0]].mergeAntenna(antennaJ)
		else:
			solutionFound = 1

	for i in range(0, len(antennas)):
		antennas[i].shrinkRange()
		antennas[i].printData()

def main():
	search([(8,0),(0,0),(4,4)],100,4)


if __name__ == "__main__":
	main()
	

