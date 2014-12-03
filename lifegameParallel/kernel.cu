
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>

#define row 30
#define column 30

using namespace std;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

//life game core
//with all 8 neighbours
__global__ void lifeGame(float *array,float *stepresult)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	//unsigned int id = threadIdx.x;
	int count = 0;
	
	//Kick off the boarder
	//test id=70
	//if (id == 70)
	//{
	//	count = array[id - column - 1] + array[id - column] + array[id - column + 1] +
	//		array[id - 1] + array[id] + array[id + 1] +
	//		array[id + column - 1] + array[id + column] + array[id + column + 1];
	//}

	//top left corner
	if (id == 0)
	{
		count = array[id] + array[id + 1] +
			array[id + column] + array[id + column + 1];
	}
	//top boarder
	else if (id < (column - 1))
	{
		count = array[id - 1] + array[id] + array[id + 1] +
			array[id + column - 1] + array[id + column] + array[id + column + 1];
	}
	//top right corner
	else if (id == (column - 1))
	{
		count = array[id - 1] + array[id] +
			array[id + column - 1] + array[id + column + 1];
	}
	//bottom left corner
	else if (id == (row - 1) * column)
	{
		count = array[id - column] + array[id - column + 1] +
			array[id] + array[id + 1];
	}
	//bottom boarder
	else if (id > (row - 1) * column && id < (row * column - 1))
	{
		count = array[id - column - 1] + array[id - column] + array[id - column + 1] +
			array[id - 1] + array[id] + array[id + 1];
	}
	//bottom right corner
	else if (id == (row*column - 1))
	{
		count = array[id - column - 1] + array[id - column] +
			array[id - 1] + array[id];
	}
	//left boarder
	else if (id % column == 0 && id != 0 && id != (row - 1)*column)
	{
		count = array[id - column] + array[id - column + 1] +
			array[id] + array[id + 1] +
			array[id + column] + array[id + column + 1];
	}
	//right boarder
	else if ((id + 1) % column == 0 && id != (column - 1) && id != (row*column - 1))
	{
		count = array[id - column - 1] + array[id - column] +
			array[id - 1] + array[id] +
			array[id + column - 1] + array[id + column];
	}
	//counting algorithm
	//This calculation is applied for cells not on boarders or corners
	//The neighborhood checks all 9 cells including itself and the surrounding neighbour in the array.
	else
	{
		count = array[id - column - 1] + array[id - column] + array[id - column + 1] +
			array[id - 1] + array[id] + array[id + 1] +
			array[id + column - 1] + array[id + column] + array[id + column + 1];
	}
	//Rules
	//The cell dies when neighbor<3 or neighbor>4.
	if (array[id] == 1 && (count < 3 || count > 4))
	{
		stepresult[id] = 0;
	}
	//The cell stays the same when neighbor=3 or =4.
	else if (array[id] == 1 && (count == 3 || count == 4))
	{
		stepresult[id] = 1;
	}
	//The cell is "born" when neighbor=3 and itself is died.
	else if (array[id] == 0 && count == 3)
	{
		stepresult[id] = 1;
	}
	else if (array[id] == 0 && count != 3)
	{
		stepresult[id] = 0;
	}
}

float initialize(float *gen)
{
	srand(time(0));
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < column; j++)
		{
			gen[i*column + j] = 0;// rand() % 2;
		}
	}
	return *gen;
}

void printResult(float *array)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < column; j++)
		{
			if (array[i*column + j] == 1)
				cout << "*";
			else
				cout << "-";
		}
		cout << endl;
	}
}

void main()
{
    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };
	
	//seed
	srand(time(0));

	//timing
	float cal_time;
	cudaEvent_t run_start,run_fin;


	//Host
	float *h_a;
	float *h_b;
	float *h_c;

	//Device
	float *d_a;
	float *d_b;

	//cuda status record
	cudaError_t cudaStatus;

	//Host mallocation
	h_a = (float*)malloc(sizeof(float)*row*column);
	h_b = (float*)malloc(sizeof(float)*row*column);
	h_c = (float*)malloc(sizeof(float)*row*column);

	//initialization
	initialize(h_a);
	initialize(h_b);
	initialize(h_c);

	for (int i = 1; i < 10;i++)
	{
		for (int j = 1; j < 10; j++)
		{
			h_a[i*column + j] = 1;
		}
	}
		

	//Device mallocation
	//life game -- array
	cudaStatus = cudaMalloc((void**)&d_a, sizeof(float)*row*column);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("\nCuda Error(cudaMalloc MatrixA):%s\n", cudaGetErrorString(cudaStatus));
		system("pause\n");
		//return 0;
	}

	//life game -- stepresult
	cudaStatus = cudaMalloc((void**)&d_b, sizeof(float)*row*column);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("\nCuda Error(cudaMalloc MatrixB):%s\n", cudaGetErrorString(cudaStatus));
		system("pause\n");
		//return 0;
	}

	//Memory copy from host to device
	cudaStatus = cudaMemcpy(d_a, h_a, sizeof(float)*row*column, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_b, h_b, sizeof(float)*row*column, cudaMemcpyHostToDevice);

	//Run kernel
	int nblocks = row*column / 512 + 1;
	//total generation times
	int counter = 0;
	while (true)
	{
		//Timer
		cudaEventCreate(&run_start);
		cudaEventCreate(&run_fin);
		cudaEventRecord(run_start, 0); //mark event
	
		//core funtion
		if (counter % 2 == 0)
		{
			lifeGame <<< nblocks, 512 >>> (d_a, d_b);
		}
		else
		{
			lifeGame <<< nblocks, 512 >> > (d_b, d_a);
		}
		
		cudaThreadSynchronize();
		cudaEventRecord(run_fin, 0);
		cudaEventSynchronize(run_fin);
		cudaEventElapsedTime(&cal_time, run_start, run_fin);

		//the resultin is in milliseconds with a resolution of around 0.5 microseconds
		printf("\n%f milliseconds passed in GPU processing\n", cal_time);

		//copy result from device to host
		cudaStatus = cudaMemcpy(h_b, d_b, sizeof(float)*row*column, cudaMemcpyDeviceToHost);
	
		//2nd time running....//

	////Memory copy from host to device
	//cudaStatus = cudaMemcpy(d_a, h_b, sizeof(float)*row*column, cudaMemcpyHostToDevice);
	//cudaStatus = cudaMemcpy(d_b, h_c, sizeof(float)*row*column, cudaMemcpyHostToDevice);


	//

	////Run kernel
	////int nblocks = row*column / 512 + 1;
	//lifeGame <<< nblocks, 512 >>> (d_a, d_b);

	//cudaThreadSynchronize();
	//


	////copy result from device to host
	//cudaStatus = cudaMemcpy(h_c, d_b, sizeof(float)*row*column, cudaMemcpyDeviceToHost);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    
	printResult(h_a);
	cout << endl;
	printResult(h_b);
	cout << endl;
	//printResult(h_c);
	system("pause");
	}
	

	cout << "all done 1" << endl;

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		//return 1;
	}

    //return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
