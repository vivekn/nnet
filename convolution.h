#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
using namespace std;

#define NUM_INPUTS 784
#define NUM_CONV 576
#define NUM_HIDDEN 50
#define NUM_CLASSES 10
#define PATCH 5
#define CDIM (29 - PATCH)

float convWeights[PATCH * PATCH];
float hiddenWeights[NUM_CONV/4][NUM_HIDDEN];
float outWeights[NUM_HIDDEN][NUM_CLASSES];

vector< vector<int> > trainImages;
vector<int> trainLabels;
vector<int> reverse_map[NUM_CONV];

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


