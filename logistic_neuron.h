#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
using namespace std;

#define NUM_PIXELS 784
#define NUM_CLASSES 10


float weights[NUM_PIXELS][NUM_CLASSES];

vector< vector<int> > trainImages;
vector<int> trainLabels;

void initialize();

float train_image(vector<int> &, int);
int classify_image(vector<int> &);

void train_data(int);
void test_data();

vector<int> threshold(vector<int>);
inline float sigmoid(float z);


