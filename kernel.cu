// Includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>

// Definitions
#define M_PI 3.14276
#define c 299792458
#define mu0 M_PI*4e-7
#define eta0 c*mu0

// GPU error checking
void checkError(cudaError cudaStatus)
{
    // [--------------- GPU error checking ---------------]
    if (cudaStatus != cudaSuccess) { // throws any errors encountered
        std::cout << stderr << " :error code: " << cudaStatus << std::endl;
        exit(1);
    }
}

// GPU kernel
__global__ void zeroesKernel(double* gpu_v1, double* gpu_v2, double* gpu_v3, double* gpu_v4, const int n) {
    // divide work amongst threads and blocks
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;
    //*/
    for (size_t i = tid + stride; i < n; i += stride) {
        gpu_v1[i] = 0;
        gpu_v2[i] = 0;
        gpu_v3[i] = 0;
        gpu_v4[i] = 0;
        __syncthreads();
    }
    //*/
} // end kern

// CPU function for source calculation
void stageSource(double* V1, double* V2, double* V3, double* V4, int x, int y, double E0, int NY) {
    /* Stage 1: Source */

    // Adapted to be 1D
    V1[x * NY + y] = V1[x * NY + y] + E0;
    V2[x * NY + y] = V2[x * NY + y] - E0;
    V3[x * NY + y] = V3[x * NY + y] - E0;
    V4[x * NY + y] = V4[x * NY + y] + E0;
    // Using 1 dimensional arrays is more obvious to work with when porting to GPU

} // end func

// CPU function
void stageScatter(double* V1, double* V2, double* V3, double* V4, int NX, int NY, double Z) {
    /* Stage 2: Scatter */
    // Variables 
    double I = 0, V = 0;

    // Parallelisable code

    // for int i = 0; i < NX*NY; i++
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            I = (2 * V1[(x * NY) + y] + 2 * V4[(x * NY) + y] - 2 * V2[(x * NY) + y] - 2 * V3[(x * NY) + y]) / (4 * Z);

            V = 2 * V1[x * NY + y] - I * Z;         //port1
            V1[x * NY + y] = V - V1[x * NY + y];

            V = 2 * V2[x * NY + y] + I * Z;         //port2
            V2[x * NY + y] = V - V2[x * NY + y];

            V = 2 * V3[x * NY + y] + I * Z;         //port3
            V3[x * NY + y] = V - V3[x * NY + y];

            V = 2 * V4[x * NY + y] - I * Z;         //port4
            V4[x * NY + y] = V - V4[x * NY + y];
        }
    }
} // end func

// GPU kernel
__global__ void scatterKernel(double* V1, double* V2, double* V3, double* V4, // Arrays
                                const int NX, const int NY, const double Z,  // Array arguments + scatter variables
                                const int Ex, const int Ey, const double E0) { // Source variables
    
        // Variables
    double I = 0, V = 0;
    // Thread identities
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Source function moved to this kernel
    if (tid == 0) {
        V1[Ex * NY + Ey] = V1[Ex * NY + Ey] + E0;
        V2[Ex * NY + Ey] = V2[Ex * NY + Ey] - E0;
        V3[Ex * NY + Ey] = V3[Ex * NY + Ey] - E0;
        V4[Ex * NY + Ey] = V4[Ex * NY + Ey] + E0;
    }
    __syncthreads();


    //*/
    for (size_t i = tid; i < NX*NY; i += stride) {
        V = 2 * V1[i] - I * Z;         //port1
        V1[i] = V - V1[i];

        V = 2 * V2[i] + I * Z;         //port2
        V2[i] = V - V2[i];

        V = 2 * V3[i] + I * Z;         //port3
        V3[i] = V - V3[i];

        V = 2 * V4[i] - I * Z;         //port4
        V4[i] = V - V4[i];
    
        __syncthreads();
    }
} // end kern

// CPU Function
void stageConnect(double* V1, double* V2, double* V3, double* V4, // Arrays
                    int NX, int NY, // Array arguments
                    double rXmin, double rXmax, double rYmin, double rYmax) { // Boundary conditions
    /* Stage 3: Connect */
    // Variables
    double tempV = 0;

    // Connect internals
    for (int x = 1; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            tempV = V2[x * NY + y];
            V2[x * NY + y] = V4[(x - 1) * NY + y];
            V4[(x - 1) * NY + y] = tempV;
        }
    }
    for (int x = 0; x < NX; x++) {
        for (int y = 1; y < NY; y++) {
            tempV = V1[x * NY + y];
            V1[x * NY + y] = V3[x * NY + y - 1];
            V3[x * NY + y - 1] = tempV;
        }
    }

    // Connect boundaries
    for (int x = 0; x < NX; x++) {
        V3[x * NY + NY - 1] = rYmax * V3[x * NY + NY - 1];
        V1[x * NY] = rYmin * V1[x * NY]; // V1[x * NY + 0] = rYmin * V1[x * NY + 0];
    }
    for (int y = 0; y < NY; y++) {
        V4[(NX - 1) * NY + y] = rXmax * V4[(NX - 1) * NY + y];
        V2[y] = rXmin * V2[y]; // V2[0 * NY + y] = rXmin * V2[0 * NY + y];
    }
} // end func


// GPU Kernel
__global__ void connectKernel(double* V1, double* V2, double* V3, double* V4, 
                                int NX, int NY, 
                            double rXmin, double rXmax, double rYmin, double rYmax) {
    /* Stage 3: Connect */
    // Variables
    double tempV = 0;
    // Thread identities
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    //*/ // Connect internals
    for (size_t i = tid + NY; i < (NX * NY); i += stride) {
        tempV = V2[i];
        V2[i] = V4[i - NY];
        V4[i - NY] = tempV;
        __syncthreads();
    }
    for (size_t i = tid + 1; i < (NX * NY); i += stride) {
        tempV = V1[i] = V3[i - 1];
        V3[i - 1] = tempV;
        __syncthreads();
    }

    //*/ // Connect boundaries
    for (size_t x = tid; x < NX; x += stride) {
        V3[x * NY + NY - 1] = rYmax * V3[x * NY + NY - 1];
        V1[x * NY] = rYmin * V1[x * NY]; // V1[x * NY + 0] = rYmin * V1[x * NY + 0];
        __syncthreads();
    }
    for (size_t y = tid; y < NY; y += stride) {
        V4[(NX - 1) * NY + y] = rXmax * V4[(NX - 1) * NY + y];
        V2[y] = rXmin * V2[y]; // V2[0 * NY + y] = rXmin * V2[0 * NY + y];
        __syncthreads();
    }

    //*/

} // end kern


int main() {

    // Start timer
    std::clock_t start = std::clock();

    /* Variables */
    // Changable variables
    int NX = 100; // number of X
    int NY = 100; // number of Y
    int NT = 10; // number of Times/Iterations
    double dl = 1;

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0); // Error checking
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0); // GPU interrogation

     // find no of blocks
    int numThreads = properties.maxThreadsPerBlock;
    int numBlocks = ((NX * NY) + numThreads - 1) / numThreads;

    double dt = dl / (sqrt(2.) * c);

    // Send to GPU
    double* gpu_v1;
    double* gpu_v2;
    double* gpu_v3;
    double* gpu_v4;
    
    // Retrieval from GPU
    //*/
    double* V1 = new double[int(NX * NY)](); // new double[int(NX*NY)](); // Sets all values to 0 
    double* V2 = new double[int(NX * NY)]();
    double* V3 = new double[int(NX * NY)](); 
    double* V4 = new double[int(NX * NY)]();
    /*/
    double* V1 = new double[int(NX * NY)]; // new double[int(NX*NY)]; // Creates array for GPU retrieval
    double* V2 = new double[int(NX * NY)];
    double* V3 = new double[int(NX * NY)]; 
    double* V4 = new double[int(NX * NY)];
    //*/

    // Scatter Coefficient
    double Z = eta0 / sqrt(2.);

    // Boundary connect Coefficiants
    double rXmin = -1;
    double rXmax = -1;
    double rYmin = -1;
    double rYmax = -1;

    // input parameters
    double width = 20 * dt * sqrt(2.);
    double delay = 100 * dt * sqrt(2.);
    int Ein[] = { 10,10 };
    // output parameters
    int Eout[] = { 15,15 };

    // file output
    std::ofstream output("GPU.csv");


    // Initialise GPU
    cudaStatus = cudaDeviceSynchronize();
    checkError(cudaStatus);

    cudaStatus = cudaMalloc((void**)&gpu_v1, (NX * NY * sizeof(double))); // Memory allocate for points array
    checkError(cudaStatus);
    cudaStatus = cudaMalloc((void**)&gpu_v2, (NX * NY * sizeof(double))); // Memory allocate for points array
    checkError(cudaStatus);
    cudaStatus = cudaMalloc((void**)&gpu_v3, (NX * NY * sizeof(double))); // Memory allocate for points array
    checkError(cudaStatus);
    cudaStatus = cudaMalloc((void**)&gpu_v4, (NX * NY * sizeof(double))); // Memory allocate for points array
    checkError(cudaStatus);

    cudaStatus = cudaDeviceSynchronize();
    checkError(cudaStatus);

    // Zero values on GPU - faster than copying array of 0's
    zeroesKernel << < numBlocks, numThreads >> > (gpu_v1, gpu_v2, gpu_v3, gpu_v4, NX*NY);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) { // throws any errors encountered
        std::cout << stderr << " :: cudaDeviceSynchronize returned error code: " << cudaStatus << std::endl;
        return cudaStatus;
    }

    for (int n = 0; n < NT; n++) {
        // Variables dependant on n
        double E0 = (1 / sqrt(2.)) * exp(-(n * dt - delay) * (n * dt - delay) / (width * width));

        
        /* Stage 1: Source */
        //stageSource(V1, V2, V3, V4, Ein[0], Ein[1], E0, NY);

        
        /* Stage 2: Scatter */
        //stageScatter(V1, V2, V3, V4, NX, NY, Z);
        scatterKernel << <numBlocks, numThreads >> > (gpu_v1, gpu_v2, gpu_v3, gpu_v4, NX, NY, Z, Ein[0], Ein[1], E0);

        
        /* Stage 3: Connect */
        //stageConnect(V1, V2, V3, V4, NX, NY, rXmin, rXmax, rYmin, rYmax);
        connectKernel << <numBlocks, numThreads >> > (gpu_v1, gpu_v2, gpu_v3, gpu_v4, NX, NY, rXmin, rXmax, rYmin, rYmax);


        /* Stage 4: Retrieval */
        //*/
        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(V1, gpu_v1, NX * NY * sizeof(double), cudaMemcpyDeviceToHost); // Memory Copy back to CPU
        if (cudaStatus != cudaSuccess) {
            std::cout << stderr << " :: cudaMemcpy failed!" << std::endl;
            return cudaStatus;
        }
        //    std::cout << "cudaMemcpy success" << std::endl;
                // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(V2, gpu_v2, NX * NY * sizeof(double), cudaMemcpyDeviceToHost); // Memory Copy back to CPU
        if (cudaStatus != cudaSuccess) {
            std::cout << stderr << " :: cudaMemcpy failed!" << std::endl;
            return cudaStatus;
        }
        //    std::cout << "cudaMemcpy success" << std::endl;
                // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(V3, gpu_v3, NX * NY * sizeof(double), cudaMemcpyDeviceToHost); // Memory Copy back to CPU
        if (cudaStatus != cudaSuccess) {
            std::cout << stderr << " :: cudaMemcpy failed!" << std::endl;
            return cudaStatus;
        }
        //    std::cout << "cudaMemcpy success" << std::endl;
                // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(V4, gpu_v4, NX * NY * sizeof(double), cudaMemcpyDeviceToHost); // Memory Copy back to CPU
        if (cudaStatus != cudaSuccess) {
            std::cout << stderr << " :: cudaMemcpy failed!" << std::endl;
            return cudaStatus;
        }
        //    std::cout << "cudaMemcpy success" << std::endl;
        //*/


        /* Stage 5: Output */

        output << n * dt << "  " << V2[Eout[0] * NY + Eout[1]] + V4[Eout[0] * NY + Eout[1]] << std::endl;
        if (n % 100 == 0)
            std::cout << n << std::endl;

    } // End of loop

    cudaFree(gpu_v1);
    cudaFree(gpu_v2);
    cudaFree(gpu_v3);
    cudaFree(gpu_v4);

    output.close();
    std::cout << "Done: " << ((std::clock() - start) / (double)CLOCKS_PER_SEC) << std::endl;
    std::cin.get();


} // end main

// EOF
