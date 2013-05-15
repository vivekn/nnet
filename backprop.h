#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
using namespace std;

#define NUM_INPUTS 784
#define NUM_HIDDEN 50
#define NUM_CLASSES 10


float hiddenWeights[NUM_INPUTS][NUM_HIDDEN];
float outWeights[NUM_HIDDEN][NUM_CLASSES];

vector< vector<int> > trainImages;
vector<int> trainLabels;

void initialize();

vector<float> propagate_hidden(vector<int> &);
vector<float> propagate_out(vector<float> &);

vector<float> error_derivs(vector<float> &, int);
vector<float> hidden_derivs(vector<float> &, vector<float> &);

float train_example(vector<int> &, int);
int classify(vector<int> &);

void train_data(int);
void test_data();

vector<int> threshold(vector<int>);
inline float sigmoid(float);


