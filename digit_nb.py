from __future__ import division
from math import log

counts = [[0 for x in xrange(784)] for y in xrange(10)]
file_counts = [0 for x in xrange(10)]
features = set()


def train_image(img, label):
	global counts, file_counts
	file_counts[label] += 1
	for i, x in enumerate(img):
		if x > 192: #thresholding
			counts[label][i] += 1

def laplace(cnt, total):
	return (cnt + 1) / (2 * total)

def classify(img):
	probs = []
	for label in xrange(10):
		prob = log(file_counts[label] / 42000)
		for i in features:
			pixel = img[i]
			if pixel > 192:
				prob += log(laplace(counts[label][i], file_counts[label]))
		probs.append((prob, label))
	p, l = max(probs)
	# print p, l, probs
	return l

def MI(pixel):
	info = 0.0
	T = sum(file_counts)
	ON = sum(counts[i][pixel] for i in xrange(10))
	OFF = T - ON

	for label in xrange(10):
		if counts[label][pixel] > 0:
			if ON > 0:
				info += counts[label][pixel] / T * log(counts[label][pixel] * T / ON / file_counts[label])
			if OFF > 0:
				info += ((file_counts[label] - counts[label][pixel]) / T * 
					log((file_counts[label] - counts[label][pixel]) * T / OFF / file_counts[label]))
	return info 

def select_features(limit=300):
	global features
	features = set(sorted(range(784), key=MI)[::-1][:limit])

def train():
	f = open("ctrain.csv")
	for line in f:
		cols = map(int, line.split())
		train_image(cols[1:], cols[0])

def test():
	f = open("ctest.csv")
	g = open("pynb.out", "w")
	for line in f:
		img = map(int, line.split())
		g.write("%d\n" % classify(img))

if __name__ == '__main__':
	train()
	select_features()
	test()
