#include "logistic_neuron.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>

static const float learnRate = 5;

void initialize() {
	// initialize weights
	for(int i=0; i < NUM_PIXELS; i++)
		for(int j=0; j < NUM_CLASSES; j++)
			weights[i][j] = (1.0 * rand() / RAND_MAX) - 0.5;
}

vector<int> threshold(vector<int> image) {
	vector<int> binaryImage(NUM_PIXELS, 0);
	for(int i=0; i < NUM_PIXELS; i++)
		binaryImage[i] = (image[i] > 127) ? 1 : 0;
	return binaryImage;
}

inline float sigmoid(float z) {
	return 1.0 / (1 + exp(-z));
}

int classify_image(vector<int> &image) {
	float labels[NUM_CLASSES];

	for(int i=0; i < NUM_CLASSES; i++) {
		float z = 0.0;
		for(int j=0; j < NUM_PIXELS; j++)
			z += weights[j][i] * image[j];
		labels[i] = sigmoid(z);
	}
	
	return max_element(labels, labels + NUM_CLASSES) - labels;
}

float train_image(vector<int> &image, int label) {
	float sq_err = 0.0;
	for(int i=0; i < NUM_CLASSES; i++) {
		float pred, z = 0.0;
		for(int j=0; j < NUM_PIXELS; j++) 
			z += weights[j][i] * image[j];
		pred = sigmoid(z);

		float target = (i == label) ? 1.0 : 0;
		float error = (target - pred);
		sq_err += (error * error);

		// Update weights by delta rule eps*(t-y)*xi*y(1-y)
		for(int j=0; j < NUM_PIXELS; j++)
			weights[j][i] += learnRate * error * image[j] * (pred * (1.0 - pred)) / NUM_PIXELS;
	}	
	return sqrt(sq_err / 10.0);
}

void train_data(int iters=50) {
	// Read data from file
	ifstream fin("ctrain.csv");

	while(!fin.eof()) {
		int label;
		vector<int> image(NUM_PIXELS, 0);

		fin >> label;
		trainLabels.push_back(label);

		for(int i=0; i < NUM_PIXELS; i++)
			fin >> image[i];

		trainImages.push_back(threshold(image));
	}

	// iterate
	for(int iter=0; iter < iters; iter++) { 
		float sumError = 0;
		for(int i=0; i < trainImages.size(); i++) 
			sumError += train_image(trainImages[i], trainLabels[i]); 
		float avgError = sumError / (trainImages.size());
		printf("%d: %f\n", iter+1, avgError);
	}
}

void test_data() {
	ifstream fin("ctest.csv");
	ofstream fout("lrnet.out");

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
