import solution1
import solution2
import time

print('Solution1.py, 100 times:')
t = time.time()
for i in range(0, 100):
	solution1.search([(23,4),(43,43),(54,54),(54,94),(24,54),(54,52),(34,23),(76,76),(87,98),(98,9),(56,6),(53,4),(23,3),(45,3),(65,4),(7,8)], 200, 1)
print(time.time() - t)

print('Solution2.py, 100 times:')
t = time.time()
for i in range(0, 100):
	solution2.search([(23,4),(43,43),(54,54),(54,94),(24,54),(54,52),(34,23),(76,76),(87,98),(98,9),(56,6),(53,4),(23,3),(45,3),(65,4),(7,8)], 200, 1)
print(time.time() - t)
