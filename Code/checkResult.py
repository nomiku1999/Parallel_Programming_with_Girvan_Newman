import sys

file1 = sys.argv[1]
file2 = sys.argv[2]

with open(file1, 'r') as f1:
	with open(file2, 'r') as f2:
		diff = set(f1).difference(f2)

diff.discard('\n')

if len(diff) == 0:
	print("Correct!")
else:
	print("Something wrong!!")
	print("Wrong line:")
	for line in diff:
		print(line)