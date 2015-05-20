#include "two_layer.h"
#include <stdlib.h>
#include <assert.h>

void initialize() {
	// initialize weights
	for(int i=0; i < NUM_PIXELS; i++)
		for(int j=0; j < NUM_CLASSES; j++)
			weights[i][j] = rand() % 1024;
}

vector<int> threshold(vector<int> image) {
	vector<int> binaryImage(NUM_PIXELS, 0);
	for(int i=0; i < NUM_PIXELS; i++)
		binaryImage[i] = (image[i] > 127) ? 1 : 0;
	return binaryImage;
}

int classify_image(vector<int> &image) {
	int labels[NUM_CLASSES] = {0};

	for(int i=0; i < NUM_PIXELS; i++) 
		for(int j=0; j < NUM_CLASSES; j++)
			labels[j] += weights[i][j] * image[i];

	return max_element(labels, labels + NUM_CLASSES) - labels;
}

void train_image(vector<int> &image, int label) {
	// Increment weights
	for(int i=0; i < NUM_PIXELS; i++)
		if (image[i] > 0)
			weights[i][label]++;

	int guess = classify_image(image);

	// Decrement weights (error correction)
	for(int i=0; i < NUM_PIXELS; i++)
		if (image[i] > 0)
			weights[i][guess]--;
}

void train_data(int iters=10) {
	// Read data from file
	ifstream fin("ctrain.csv");

	while(!fin.eof()) {
		int label;
		vector<int> image(NUM_PIXELS, 0);

		fin >> label;
		trainLabels.push_back(label);

		for(int i=0; i < NUM_PIXELS; i++)
			fin >> image[i];

		trainImages.push_back(image);
	}

	// iterate
	for(int iter=0; iter < iters; iter++) 
		for(int i=0; i < trainImages.size(); i++) {
			vector<int> binaryImage = threshold(trainImages[i]);
			train_image(binaryImage, trainLabels[i]);
		}
}

void test_data() {
	ifstream fin("ctest.csv");
	ofstream fout("test-output");

	while(!fin.eof()) {
		vector<int> image(NUM_PIXELS, 0);

		bool flag = false;

		for(int i=0; i < NUM_PIXELS; i++) {
			fin >> image[i];
			if (fin.eof()) {
				flag = true;
				break;
			}
		}

		if (flag) break;


		vector<int> binaryImage = threshold(image);

		fout << classify_image(binaryImage) << endl;
		// cout << image[123] << " "<< classify_image(binaryImage) << endl;
	}
}

int main() {
	initialize();
	train_data();
	test_data();
}
