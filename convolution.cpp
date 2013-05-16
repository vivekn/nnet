#include "convolution.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#define all(v) v.begin(), v.end()
#define EPS 1e-18

static float alpha0 = 0.1, alpha;
int ctr = 0;

void initialize() {
	// initialize weights
	for(int i=0; i < NUM_CONV/4; i++)
		for(int j=0; j < NUM_HIDDEN; j++)
			hiddenWeights[i][j] = (1.0 * rand() / RAND_MAX) - 0.5;

	for(int i=0; i < NUM_HIDDEN; i++)
		for(int j=0; j < NUM_CLASSES; j++)
			outWeights[i][j] = (1.0 * rand() / RAND_MAX) - 0.5;

	for(int i=0; i < PATCH; i++)
		for(int j=0; j < PATCH; j++)
			convWeights[i*PATCH + j] = (1.0 * rand() / RAND_MAX) - 0.5;

	build_maps();
}

void build_maps() {
	for(int i=0; i < NUM_CONV; i++) {
		int x = i / CDIM, y = i % CDIM;
		for(int dx=0; dx < PATCH; dx++)
			for(int dy=0; dy < PATCH; dy++)
				reverse_map[i].push_back((x+dx) * 28 + (y+dy));
	}
}

vector<int> threshold(vector<int> image) {
	vector<int> binaryImage(NUM_INPUTS, 0);
	for(int i=0; i < NUM_INPUTS; i++)
		binaryImage[i] = (image[i] > 127) ? 1 : 0;
	return binaryImage;
}

vector<float> downsample(vector<float> image) {
	vector<float> result(NUM_CONV/4, 0);
	for(int i=0; i < CDIM; i+=2)
		for(int j=0; j < CDIM; j+=2) {
			float pixels[] = {image[i*CDIM + j], image[(i+1)* CDIM + j], image[i*CDIM + j + 1], image[(i+1)*CDIM + j+1]};
			result[i/2*(CDIM/2) + j/2] = *max_element(pixels, pixels+4);
		}
	return result;
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

vector<float> convolute(vector<int> &image) {
	vector<float> result(NUM_CONV, 0);

	for(int i=0; i < NUM_CONV; i++) {
		float z = 0.0;
		for(int j=0; j < reverse_map[i].size(); j++) 
			z += convWeights[j]*image[reverse_map[i][j]];
		result[i] = sigmoid(z);
	}

	return result;
}

vector<float> propagate_hidden(vector<float> &inputs) {
	vector<float> result(NUM_HIDDEN, 0);
	for(int i=0; i < NUM_HIDDEN; i++) {
		float z = 0.0;
		for(int j=0; j < NUM_CONV/4; j++)
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
		result[i] = (t - outActivity[i]);
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

vector<float> image_derivs(vector<float> &convImage, vector<float> &hderivs) {
	vector<float> result(NUM_CONV/4, 0);

	for(int i=0; i < NUM_CONV/4; i++)
		for(int j=0; j < NUM_HIDDEN; j++) 
			result[i] += hiddenWeights[i][j] * convImage[i] * (1.0 - convImage[i]) * hderivs[j];

	return result;
}

int classify(vector<int> &image) {
	vector<float> conv = downsample(convolute(image));
	vector<float> hact = propagate_hidden(conv);
	vector<float> probs = propagate_out(hact);
	return max_element(all(probs)) - probs.begin();	
}

float train_example(vector<int> &image, int label) {
	vector<float> conv = downsample(convolute(image));
	vector<float> hact = propagate_hidden(conv);
	vector<float> probs = propagate_out(hact);

	vector<float> oderivs = error_derivs(probs, label);
	vector<float> hderivs = hidden_derivs(hact, oderivs);
	vector<float> iderivs = image_derivs(conv, hderivs);
	
	// update output layer weights
	for(int i=0; i < NUM_HIDDEN; i++)
		for(int j=0; j < NUM_CLASSES; j++)
			outWeights[i][j] += alpha * hact[i] * oderivs[j];

	// update hidden layer weights
	for(int i=0; i < NUM_CONV/4; i++)
		for(int j=0; j < NUM_HIDDEN; j++)
			hiddenWeights[i][j] += alpha * conv[i] * hderivs[j];

	// Update image layer weights
	for(int i=0; i < NUM_CONV; i++) {
		int x = (i / CDIM) / 2, y = (i % CDIM) / 2;
		int index = x * CDIM / 2 + y * CDIM / 2; // because of downsampling

		for(int j=0; j < reverse_map[i].size(); j++)
			convWeights[j] += alpha * image[reverse_map[i][j]] * iderivs[index];
	}

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

void train_data(int iters=5) {
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
		alpha = alpha0 / (1.0 + (iter / 6.0));
		for(int i=0; i < trainImages.size(); i++) 
			sumError += train_example(trainImages[i], trainLabels[i]);
		float avgError = sumError / trainImages.size();
		printf("%d. Error: %f\n", iter, avgError);
	}
}

void test_data() {
	ifstream fin("ctest.csv");
	ofstream fout("cnn.out");

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
