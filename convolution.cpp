#include "convolution.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#define all(v) v.begin(), v.end()
#define EPS 1e-18

static float alpha0 = 0.05, alpha, momentum = 0.5;
int ctr = 0;

void initialize() {
	// initialize weights
	for(int i=0; i < NUM_CONV; i++)
		for(int j=0; j < NUM_HIDDEN; j++)
			hiddenWeights[i][j] = (1.0 * rand() / RAND_MAX) - 0.5;

	for(int i=0; i < NUM_HIDDEN; i++)
		for(int j=0; j < NUM_CLASSES; j++)
			outWeights[i][j] = (1.0 * rand() / RAND_MAX) - 0.5;


	for(int i=0; i < PATCH; i++)
		for(int j=0; j < PATCH; j++)
			for(int k=0; k < NFILTERS; k++)
				convWeights[k][i*PATCH + j] = (1.0 * rand() / RAND_MAX) - 0.5;

	build_maps();
}

void build_maps() {
	for(int k=0; k < NFILTERS; k++)
	for(int i=0; i < NUM_CONV; i++) {
		int x = i / CDIM, y = i % CDIM;
		for(int dx=0; dx < PATCH; dx++)
			for(int dy=0; dy < PATCH; dy++)
				reverse_map[k * NUM_CONV + i].push_back((x+dx) * 28 + (y+dy));
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
	vector<float> result(NFILTERS * NUM_CONV, 0);

	for(int k=0; k < NFILTERS; k++)
	for(int i=0; i < NUM_CONV; i++) {
		float z = 0.0;
		for(int j=0; j < reverse_map[k*NUM_CONV+i].size(); j++) 
			z += convWeights[k][j]*image[reverse_map[k*NUM_CONV+i][j]];
		result[k*NUM_CONV+i] = sigmoid(z);
	}

	return result;
}

vector<float> propagate_hidden(vector<float> &inputs) {
	vector<float> result(NUM_HIDDEN, 0);
	for(int i=0; i < NUM_HIDDEN; i++) {
		float z = 0.0;
		for(int j=0; j < NUM_CONV; j++)
			for(int k=0; k < NFILTERS; k++)
				z += inputs[k*NUM_CONV+j] * hiddenWeights[k*NUM_CONV+j][i];
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
	vector<float> result(NUM_CONV*NFILTERS, 0);

	for(int i=0; i < NUM_CONV; i++)
		for(int k=0; k < NFILTERS; k++)
			for(int j=0; j < NUM_HIDDEN; j++) 
				result[k*NUM_CONV+i] += hiddenWeights[k*NUM_CONV+i][j] * convImage[k*NUM_CONV+i] * (1.0 - convImage[k*NUM_CONV+i]) * hderivs[j];

	return result;
}

int classify(vector<int> &image) {
	vector<float> conv = (convolute(image));
	vector<float> hact = propagate_hidden(conv);
	vector<float> probs = propagate_out(hact);
	return max_element(all(probs)) - probs.begin();	
}

float train_example(vector<int> &image, int label) {
	vector<float> conv = (convolute(image));
	vector<float> hact = propagate_hidden(conv);
	vector<float> probs = propagate_out(hact);

	vector<float> oderivs = error_derivs(probs, label);
	vector<float> hderivs = hidden_derivs(hact, oderivs);
	vector<float> iderivs = image_derivs(conv, hderivs);
	
	float delta;
	// update output layer weights
	for(int i=0; i < NUM_HIDDEN; i++)
		for(int j=0; j < NUM_CLASSES; j++){
			delta = alpha * hact[i] * oderivs[j];
			outVelocity[i][j] = outVelocity[i][j]*momentum  + delta;
			outWeights[i][j] += outVelocity[i][j];
		}
	// update hidden layer weights
	for(int i=0; i < NUM_CONV; i++)
		for(int j=0; j < NUM_HIDDEN; j++)
			for(int k=0; k < NFILTERS; k++){
				delta = alpha * conv[k*NUM_CONV+i] * hderivs[j];
				hiddenVelocity[k*NUM_CONV+i][j] = hiddenVelocity[k*NUM_CONV+i][j]*momentum + delta;
				hiddenWeights[k*NUM_CONV+i][j] += hiddenVelocity[k*NUM_CONV+i][j];
			}
	// Update image layer weights
	for(int i=0; i < NUM_CONV; i++) {
		// int x = (i / CDIM) / 2, y = (i % CDIM) / 2;
		int index = i; // x * CDIM / 2 + y * CDIM / 2; // because of downsampling

		for(int j=0; j < reverse_map[i].size(); j++)
			for(int k=0; k < NFILTERS; k++) {
				delta = alpha * image[reverse_map[i+k*NUM_CONV][j]] * iderivs[i+k*NUM_CONV];
				convVelocity[k][j] = momentum*convVelocity[k][j] + delta;
				convWeights[k][j] += convVelocity[k][j];

			}
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

void train_data(int iters=10) {
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

void train_batch(int iters=5, int num_batches=1) {
	// Read data from file
	ifstream fin("minibatch.csv");

	while(!fin.eof()) {
		int label;
		vector<int> image(NUM_INPUTS, 0);

		fin >> label;
		trainLabels.push_back(label);

		for(int i=0; i < NUM_INPUTS; i++){
			fin >> image[i];
			image[i] -= 128;
		}

		trainImages.push_back((image));
	}

	// iterate
	for(int batch=0; batch < num_batches; batch++)
		for(int iter=0; iter < iters; iter++) { 
			int giter = iter + (batch * iters);
			alpha = alpha0;
			// alpha = alpha0 / (1.0 + (giter / 6.0));
			float sumError = 0;
			for(int i=batch * BATCH_SIZE; i < (batch + 1) * BATCH_SIZE; i++) 
				sumError += train_example(trainImages[i], trainLabels[i]);
			float avgError = sumError / trainImages.size();
			printf("%d. Error: %f\n", iter, avgError);
		}
}

void test_data() {
	ifstream fin("ctest.csv");
	ofstream fout("cnn2.out");

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


		vector<int> binaryImage = (image);

		fout << classify(binaryImage) << endl;
		// cout << image[123] << " "<< classify_image(binaryImage) << endl;
	}
}

int main() {
	initialize();
	train_batch();
	test_data();
}
