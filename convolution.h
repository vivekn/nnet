#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
using namespace std;

#define NUM_INPUTS 784
#define NUM_CONV 484
#define NUM_HIDDEN 50
#define NUM_CLASSES 10
#define PATCH 7
#define CDIM (29 - PATCH)
#define NFILTERS 1
#define BATCH_SIZE 42000

float convWeights[NFILTERS][PATCH * PATCH];
float hiddenWeights[NFILTERS*NUM_CONV][NUM_HIDDEN];
float outWeights[NUM_HIDDEN][NUM_CLASSES];

float convVelocity[NFILTERS][PATCH * PATCH] = {0};
float hiddenVelocity[NFILTERS*NUM_CONV][NUM_HIDDEN] = {0};
float outVelocity[NUM_HIDDEN][NUM_CLASSES] = {0};

vector< vector<int> > trainImages;
vector<int> trainLabels;
vector<int> reverse_map[NFILTERS*NUM_CONV];

void initialize();

void build_maps();

vector<float> convolute(vector<int> &);
vector<float> propagate_hidden(vector<float> &);
vector<float> propagate_out(vector<float> &);

vector<float> error_derivs(vector<float> &, int);
vector<float> hidden_derivs(vector<float> &, vector<float> &);
vector<float> image_derivs(vector<float> &, vector<float> &);

float train_example(vector<int> &, int);
int classify(vector<int> &);

void train_data(int);
void test_data();

vector<int> threshold(vector<int>);
vector<float> downsample(vector<float>);
inline float sigmoid(float);


