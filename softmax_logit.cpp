#include "softmax_logit.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#define EPS 1e-18

static const double learnRate = 0.01;

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

inline double sigmoid(double z) {
	return 1.0 / (1 + exp(-z));
}

int classify_image(vector<int> &image) {
	double logits[NUM_CLASSES];

	for(int i=0; i < NUM_CLASSES; i++) {
		double z = 0.0;
		for(int j=0; j < NUM_PIXELS; j++)
			z += weights[j][i] * image[j];
		logits[i] = z;
	}
	
	return max_element(logits, logits + NUM_CLASSES) - logits;
}

double train_image(vector<int> &image, int label) {
	double logits[NUM_CLASSES], softmax_den = 0;

	for(int i=0; i < NUM_CLASSES; i++) {
		double z = 0.0;
		for(int j=0; j < NUM_PIXELS; j++) {
			z += weights[j][i] * image[j];
		}
		logits[i] = z;
	}

	double K = *max_element(logits, logits + NUM_CLASSES); // Prevents overflow

	for(int i=0; i < NUM_CLASSES; i++) {
		logits[i] = exp(logits[i] - K) + EPS;
		softmax_den += logits[i];
	}
	
	double C = 0.0;

	for(int i=0; i < NUM_CLASSES; i++) {
		double pred = logits[i] / softmax_den;
		double t = (i == label) ? 1.0 : 0.0;
		for(int j=0; j < NUM_PIXELS; j++) 
			weights[j][i] -= learnRate * image[j] * (pred - t);

		if (t > 0)
			C -= t * log(pred);
	}	
	// printf("C = %lf\n", C);
	return C;
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
	puts("done");

	// iterate
	for(int iter=0; iter < iters; iter++) { 
		double sumError = 0;
		for(int i=0; i < trainImages.size(); i++) 
			sumError += train_image(trainImages[i], trainLabels[i]); 
		double avgError = sumError / (trainImages.size());
		printf("%d: X entropy = %f\n", iter+1, avgError);
	}
}

void test_data() {
	ifstream fin("ctest.csv");
	ofstream fout("smlrnet.out");

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
