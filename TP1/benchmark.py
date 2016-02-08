import solution1
import solution2
import time
import random

nbPoles = 300
rangePoles = 1000
nbIterations = 2
costConstant = 200
costRange = 1

polesList = []

for i in range(0, nbPoles):
	polesList.append((random.randint(0, rangePoles), random.randint(0, rangePoles)))

print('Solution1.py, ' + str(nbIterations) + ' times:')
t = time.time()
for i in range(0, nbIterations):
	solution1.search(list(polesList), costConstant, costRange)
print(time.time() - t)

print('Solution2.py, ' + str(nbIterations) + ' times:')
t = time.time()
for i in range(0, nbIterations):
	solution2.search(list(polesList), costConstant, costRange)
print(time.time() - t)
