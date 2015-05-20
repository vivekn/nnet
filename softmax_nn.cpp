#include "backprop.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#define all(v) v.begin(), v.end()
#define EPS 1e-18

static float alpha0 = 0.1, alpha;
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

void softmax(vector<float> &result) {
	float K = *max_element(all(result));
	float denom = 0;

	for(int i=0; i < result.size(); i++) {
		result[i] = exp(result[i] - K) + EPS;
		denom += result[i];
	}

	for(int i=0; i < result.size(); i++) 
		result[i] /= denom;

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
		result[i] = z;
	}
	softmax(result);
	return result;
}

vector<float> error_derivs(vector<float> &outActivity, int label) {
	vector<float> result(NUM_CLASSES, 0);
	for(int i=0; i < NUM_CLASSES; i++) {
		float t = (i == label) ? 1.0 : 0;
		result[i] = outActivity[i] - t;
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
			outWeights[i][j] -= alpha * hact[i] * oderivs[j];

	// update hidden layer weights
	for(int i=0; i < NUM_INPUTS; i++)
		for(int j=0; j < NUM_HIDDEN; j++)
			hiddenWeights[i][j] -= alpha * image[i] * hderivs[j];


	// check progress
	double C = 0.0;

	for(int i=0; i < NUM_CLASSES; i++) {
		double pred = probs[i];
		double t = (i == label) ? 1.0 : 0.0;
		if (t > 0)
			C -= t * log(pred);
	}	
	return C;
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
		alpha = alpha0 / (1.0 + iter/4.0);
		float sumError = 0;
		for(int i=0; i < trainImages.size(); i++) 
			sumError += train_example(trainImages[i], trainLabels[i]);
		float avgError = sumError / trainImages.size();
		printf("iter %d, X entropy: %f\n", iter+1, avgError);
	}
}

void test_data() {
	ifstream fin("ctest.csv");
	ofstream fout("snn.out");

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
