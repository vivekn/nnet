#include "backprop.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#define all(v) v.begin(), v.end()

static const float alpha = 0.1;
int ctr = 0;

void initialize() {
	// initialize weights
	for(int i=0; i < NUM_INPUTS; i++)
		for(int j=0; j < NUM_HIDDEN; j++)
			hiddenWeights[i][j] = (1.0 * rand() / RAND_MAX) - 0.5;

	for(int i=0; i < NUM_HIDDEN; i++)
		for(int j=0; j < NUM_CLASSES; j++)
			outWeights[i][j] = (1.0 * rand() / RAND_MAX) - 0.5;
}

vector<int> threshold(vector<int> image) {
	vector<int> binaryImage(NUM_INPUTS, 0);
	for(int i=0; i < NUM_INPUTS; i++)
		binaryImage[i] = (image[i] > 127) ? 1 : 0;
	return binaryImage;
}

inline float sigmoid(float z) {
	return 1.0 / (1 + exp(-z));
}

vector<float> propagate_hidden(vector<int> &inputs) {
	vector<float> result(NUM_HIDDEN, 0);
	for(int i=0; i < NUM_HIDDEN; i++) {
		float z = 0.0;
		for(int j=0; j < NUM_INPUTS; j++)
			z += inputs[j] * hiddenWeights[j][i];
		result[i] = sigmoid(z);
	}
	return result;
}

vector<float> propagate_out(vector<float> &hiddenActivity) {
	vector<float> result(NUM_CLASSES, 0);
	for(int i=0; i < NUM_CLASSES; i++) {
		float z = 0.0;
		for(int j=0; j < NUM_HIDDEN; j++)
			z += hiddenActivity[j] * outWeights[j][i];
		result[i] = sigmoid(z);
	}
	return result;
}

vector<float> error_derivs(vector<float> &outActivity, int label) {
	vector<float> result(NUM_CLASSES, 0);
	for(int i=0; i < NUM_CLASSES; i++) {
		float t = (i == label) ? 1.0 : 0;
		result[i] = (t - outActivity[i]) * (outActivity[i]) * (1.0 - outActivity[i]);
	}
	return result;
}

vector<float> hidden_derivs(vector<float> &hiddenActivity, vector<float> &oderivs) {
	vector<float> result(NUM_HIDDEN, 0);
	
	for(int i=0; i < NUM_HIDDEN; i++) 
		for(int j=0; j < NUM_CLASSES; j++)
			result[i] += outWeights[i][j] * hiddenActivity[i] * (1.0 - hiddenActivity[i]) * oderivs[j]; 
	
	return result;
}

int classify(vector<int> &image) {
	vector<float> hact = propagate_hidden(image);
	vector<float> probs = propagate_out(hact);
	return max_element(all(probs)) - probs.begin();	
}

float train_example(vector<int> &image, int label) {
	vector<float> hact = propagate_hidden(image);
	vector<float> probs = propagate_out(hact);
	vector<float> oderivs = error_derivs(probs, label);
	vector<float> hderivs = hidden_derivs(hact, oderivs);

	// update output layer weights
	for(int i=0; i < NUM_HIDDEN; i++)
		for(int j=0; j < NUM_CLASSES; j++)
			outWeights[i][j] += alpha * hact[i] * oderivs[j];

	// update hidden layer weights
	for(int i=0; i < NUM_INPUTS; i++)
		for(int j=0; j < NUM_HIDDEN; j++)
			hiddenWeights[i][j] += alpha * image[i] * hderivs[j];


	// check progress
	float mse = 0.0;

	for(int i=0; i < NUM_CLASSES; i++) {
		float t = (i == label) ? 1.0 : 0.0;
		mse += (t-probs[i]) * (t - probs[i]);
	}
	mse = sqrt(mse / (1.0 * NUM_CLASSES));
	return mse;
}

void train_data(int iters=20) {
	// Read data from file
	ifstream fin("ctrain.csv");

	while(!fin.eof()) {
		int label;
		vector<int> image(NUM_INPUTS, 0);

		fin >> label;
		trainLabels.push_back(label);

		for(int i=0; i < NUM_INPUTS; i++)
			fin >> image[i];

		trainImages.push_back(threshold(image));
	}

	// iterate
	for(int iter=0; iter < iters; iter++) { 
		float sumError = 0;
		for(int i=0; i < trainImages.size(); i++) 
			sumError += train_example(trainImages[i], trainLabels[i]);
		float avgError = sumError / trainImages.size();
		printf("%d. Error: %f\n", iter, avgError);
	}
}

void test_data() {
	ifstream fin("ctest.csv");
	ofstream fout("bnet.out");

	while(!fin.eof()) {
		vector<int> image(NUM_INPUTS, 0);

		bool flag = false;

		for(int i=0; i < NUM_INPUTS; i++) {
			fin >> image[i];
			if (fin.eof()) {
				flag = true;
				break;
			}
		}

		if (flag) break;


		vector<int> binaryImage = threshold(image);

		fout << classify(binaryImage) << endl;
		// cout << image[123] << " "<< classify_image(binaryImage) << endl;
	}
}

int main() {
	initialize();
	train_data();
	test_data();
}
