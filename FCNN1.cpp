// FCNN1.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include <vector>
#include <ctime>

#define USE_MNIST_LOADER
#define MNIST_FLOAT
#include "MNIST.h"

using namespace std;

float first_proizv(float v)
{
	return 1.0 / (cosh(v)*cosh(v));
}

float f1(float v)
{
	return tanh(v);
}

int main()
{
	mnist_data *train;
	mnist_data *test;
	unsigned int cntTrain;
	unsigned int cntTest;
	int eq = 0.0;

	mnist_load("F:/ВУЗ/deep learning/MNIST/train-images.idx3-ubyte", "F:/ВУЗ/deep learning/MNIST/train-labels.idx1-ubyte", &train, &cntTrain);
	mnist_load("F:/ВУЗ/deep learning/MNIST/t10k-images.idx3-ubyte", "F:/ВУЗ/deep learning/MNIST/t10k-labels.idx1-ubyte", &test, &cntTest);

	int m = 10;
	int k = 10;
	int n = 28 * 28;
	int era = 10;
	float eta = 0.01;

	float *g = new float[m];
	float *f = new float[k+1];
	float *f_pr = new float[k+1];
	float *v = new float[k+1];
	float *w1 = new float[(k + 1)*(n + 1)];
	float *w2 = new float[(k + 1)*m];
	float *e_g = new float[m];
	float sum_e = 0.0;
	float *u = new float[m];
	float *dE = new float[(k + 1)*(n + 1) + (k + 1)*m];
	vector<int> index;

	//initialize w1,w2
	srand(time(0));
	for (int s = 0; s < (k + 1)*(n+1); s++)
	{
		w1[s] = ((float)(-100 + rand() % 200) / 100.0) *(2.0 / (float)(n + m));
	}

	for (int s = 0; s < (k + 1)*m; s++)
	{
		w2[s] = ((float)(-100 + rand() % 200) / 100.0) *(2.0 / (float)(m + k));
	}

	for (int z = 0; z < era; z++)
	{
		for (int i = 0; i < cntTrain; i++)
		{
			index.push_back(i);
		}
		random_shuffle(index.begin(), index.end());

		for (int l = 0; l < cntTrain; l++)
		{
			//step1
			#pragma omp parallel for
			for (int s = 0; s < k+1; s++)
			{
				f[s] = w1[s] * train[index.at(l)].data[0];
				for (int i = 1; i < n+1; i++)
				{
					f[s] += w1[(i)*(k+1)+s] * train[index.at(l)].data[i];
				}
				
				f_pr[s] = first_proizv(f[s]);

				v[s] = f1(f[s]);
			}
			v[0] = 1.0;
			f_pr[0] = 0.0;

			#pragma omp parallel for
			for (int j = 0; j < m; j++)
			{
				g[j] = 0.0;
				for (int s = 0; s < k + 1; s++)
				{
					g[j] += w2[s*m+j] * v[s];
				}

				e_g[j] = expf(g[j]);
			}

			//step2
			sum_e = 0.0;
			for (int j = 0; j < m; j++)
			{
				sum_e += e_g[j];
			}

			#pragma omp parallel for
			for (int j = 0; j < m; j++)
			{
				u[j] = e_g[j] / sum_e;
			}

			//step3 for w1
			#pragma omp parallel for
			for (int s = 0; s < k + 1; s++)
			{
				for (int i = 0; i < n+1; i++)
				{
					dE[i*(k + 1) + s] = 0.0;
					for (int j = 0; j < m; j++)
					{
						if (train[index.at(l)].label_mas[j] == 0) continue;
						dE[i*(k + 1) + s] -= (float)train[index.at(l)].label_mas[j] * ((sum_e - e_g[j]) / sum_e) * w2[s*m + j] * f_pr[s] * train[index.at(l)].data[i];
					}
				}
			}

			//step4 for w2
			#pragma omp parallel for
			for (int j = 0; j < m; j++)
			{
				for (int s = 0; s < k+1; s++)
				{
					dE[(k + 1)*(n + 1) +s*m+j] = 0.0;
					for (int jj = 0; jj < m; jj++)
					{
						if (train[index.at(l)].label_mas[jj] == 0||j==jj) continue;
						dE[(k + 1)*(n + 1) + s*m + j] += (float)train[index.at(l)].label_mas[jj] * (e_g[j] / sum_e) * v[s];
					}
					dE[(k + 1)*(n + 1) + s*m + j] -= (float)train[index.at(l)].label_mas[j] * ((sum_e - e_g[j]) / sum_e) * v[s];
				}
			}

			//step5
			#pragma omp parallel for
			for (int s = 0; s < (k + 1)*(n + 1); s++)
				w1[s] -= eta*dE[s];
			
			#pragma omp parallel for
			for (int s = 0; s < (k + 1)*m; s++)
				w2[s] -= eta*dE[(k + 1)*(n + 1) + s];
		}
		cout << "era: " << z << endl;
		index.clear();
	}


	//test
	for (int z = 0; z < cntTest; z++)
	{
		#pragma omp parallel for
		for (int s = 0; s < k + 1; s++)
		{
			f[s] = w1[s] * test[z].data[0];
			for (int i = 1; i < n+1; i++)
			{
				f[s] += w1[(i)*(k + 1) + s] * test[z].data[i];
			}

			v[s] = f1(f[s]);
		}
		v[0] = 1.0;

		#pragma omp parallel for
		for (int j = 0; j < m; j++)
		{
			g[j] = 0.0;
			for (int s = 0; s < k + 1; s++)
			{
				g[j] += w2[s*m + j] * v[s];
			}
			e_g[j] = expf(g[j]);
		}

		sum_e = 0.0;
		for (int j = 0; j < m; j++)
		{
			sum_e += e_g[j];
		}

		float max = -1;
		int index_max = -1;
		for (int j = 0; j < m; j++)
		{
			u[j] = e_g[j] / sum_e;
			if (max < u[j])
			{
				max = u[j];
				index_max = j;
			}
		}
		if (test[z].label == index_max) eq++;
	}

	cout << "accuracy: " << ((float)eq / (float)(cntTest))*100.0 << endl;
	
	delete[] g;
	delete[] f;
	delete[] f_pr;
	delete[] v;
	delete[] w1;
	delete[] w2;
	delete[] e_g;
	delete[] u;
	delete[] dE;
	delete test;
	delete train;
	
	system("pause");
	return 0;
}

