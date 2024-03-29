// Includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>

//#define M_PI 3.14276 // unused
#define c 299792458
//#define mu0 M_PI*4e-7 // unused
//#define eta0 c*mu0 // unused

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


// GPU kernel
__global__ void scatterKernel(double* gpu_v1, double* gpu_v2, double* gpu_v3, double* gpu_v4, // Arrays
                                const int NX, const int NY,  // Array arguments
                                const int Ex, const int Ey, const double E0) { // Source function variables

        // Variables
    double V = 0; // V is a temp variable
    // Thread identities
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    /* Stage 1: Source */

    // Source function moved to this kernel, 
    if (tid == 0) gpu_v1[Ex * NY + Ey] = gpu_v1[Ex * NY + Ey] + E0;
    if (tid == 1) gpu_v2[Ex * NY + Ey] = gpu_v2[Ex * NY + Ey] - E0;
    if (tid == 2) gpu_v3[Ex * NY + Ey] = gpu_v3[Ex * NY + Ey] - E0;
    if (tid == 3) gpu_v4[Ex * NY + Ey] = gpu_v4[Ex * NY + Ey] + E0;
    __syncthreads();


    /* Stage 2: Scatter */

    //*/
    for (size_t i = tid; i < NX*NY; i += stride) {
        // Tidied up
        double I = ((gpu_v1[i] + gpu_v4[i] - gpu_v2[i] - gpu_v3[i]) / 2); // Calculate coefficient
        //I = (2 * V1[(x * NY) + y] + 2 * V4[(x * NY) + y] - 2 * V2[(x * NY) + y] - 2 * V3[(x * NY) + y]) / (4 * Z);

        V = 2 * gpu_v1[i] - I;         //port1
        gpu_v1[i] = V - gpu_v1[i];

        V = 2 * gpu_v2[i] + I;         //port2
        gpu_v2[i] = V - gpu_v2[i];

        V = 2 * gpu_v3[i] + I;         //port3
        gpu_v3[i] = V - gpu_v3[i];

        V = 2 * gpu_v4[i] - I;         //port4
        gpu_v4[i] = V - gpu_v4[i];
    
        __syncthreads();
    }
} // end kern


// GPU Kernel
__global__ void connectKernel(double* gpu_v1, double* gpu_v2, double* gpu_v3, double* gpu_v4, double* gpu_out, // arrays
    const int NX, const int NY, const int n, const int Ex, const int Ey, // array variables
    const double rXmin, const double rXmax, const double rYmin, const double rYmax) { // boundary variables

/* Stage 3: Connect */

    // Variables
    double tempV = 0; // temporary variable for swapping values

    // Thread identities
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Connect ports 2 and 4
    for (size_t i = (tid + NY); i < (NX * NY); i += stride) { // Loop only through nodes where X > 0
        tempV = gpu_v2[i];
        gpu_v2[i] = gpu_v4[i - NY];
        gpu_v4[i - NY] = tempV;
    }
    __syncthreads(); // Sync between loops

    // Connect ports 1 and 3
    for (size_t i = tid + 1; i < (NX * NY); i += stride) { // Loop only through nodes where Y > 0
        // Skip when finding y = 0
        if (i % NY != 0) {
            tempV = gpu_v1[i];
            gpu_v1[i] = gpu_v3[i - 1];
            gpu_v3[i - 1] = tempV;
        }
    }
    __syncthreads(); // Sync

    // Connect boundaries
    for (size_t x = tid; x < NX; x += stride) {
        gpu_v3[x * NY + NY - 1] = rYmax * gpu_v3[x * NY + NY - 1];
        gpu_v1[x * NY] = rYmin * gpu_v1[x * NY]; // V1[x * NY + 0] = rYmin * V1[x * NY + 0];
    }
    __syncthreads(); // Sync between loops
    for (size_t y = tid; y < NY; y += stride) {
        gpu_v4[(NX - 1) * NY + y] = rXmax * gpu_v4[(NX - 1) * NY + y];
        gpu_v2[y] = rXmin * gpu_v2[y]; // V2[0 * NY + y] = rXmin * V2[0 * NY + y];
    }
    __syncthreads(); // Sync between loops

    /* Stage 4: Output */
    if (tid == 0) {
        gpu_out[n] = gpu_v2[Ex * NY + Ey] + gpu_v4[Ex * NY + Ey];
    }

} // end kern


int main() {

    // Start timer
    std::clock_t start = std::clock();

    /* Stage 0: Setup and Variables */

    // Changable variables
    int NX = 100; // number of X
    int NY = 100; // number of Y
    int NT = 8192; // number of Times/Iterations
    double dl = 1;

    // CUDA/ GPU dependant variables
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0); // Error checking
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0); // GPU interrogation

     // find no of blocks
    int numThreads = properties.maxThreadsPerBlock;
    int numBlocks = ((NX * NY) + numThreads - 1) / numThreads;

    double dt = dl / (sqrt(2.) * c); // Time step size

    // Send to GPU
    double* gpu_v1; // 
    double* gpu_v2; // 
    double* gpu_v3; //
    double* gpu_v4; // > Arrays for data points
    double* gpu_out;
    
    // Retrieval from GPU
    // Not required to be stored on CPU
    /*/
    double* V1 = new double[int(NX * NY)]; // Creates array for GPU retrieval
    double* V2 = new double[int(NX * NY)];
    double* V3 = new double[int(NX * NY)]; 
    double* V4 = new double[int(NX * NY)];
    //*/
    double* h_out = new double[NT](); // initialise and set to 0

    // Scatter Coefficient -- unused
    //double Z = eta0 / sqrt(2.);

    // Boundary connect Coefficiants
    double rXmin = -1;
    double rXmax = -1;
    double rYmin = -1;
    double rYmax = -1;

    // input parameters
    double width = 20 * dt * sqrt(2.); // width of impulse
    double delay = 100 * dt * sqrt(2.); // delay before impulse
    int Ein[] = { 10,10 }; // input position
    // output parameters
    int Eout[] = { 15,15 }; // read position

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
    cudaStatus = cudaMalloc((void**)&gpu_out, (NT * sizeof(double))); // Memory allocate for results array
    checkError(cudaStatus);

    cudaStatus = cudaDeviceSynchronize(); // Synchronise device before running kernels
    checkError(cudaStatus);

    // Zero values on GPU - faster than copying array of 0's
    zeroesKernel << < numBlocks, numThreads >> > (gpu_v1, gpu_v2, gpu_v3, gpu_v4, NX*NY);
    cudaStatus = cudaDeviceSynchronize();
    checkError(cudaStatus);

    for (int n = 0; n < NT; n++) {
        // Variables dependant on n
        double E0 = (1 / sqrt(2.)) * exp(-(n * dt - delay) * (n * dt - delay) / (width * width)); // Finds where impulse is in time

        /* Stages 1 and 2 */
        scatterKernel << <numBlocks, numThreads >> > (gpu_v1, gpu_v2, gpu_v3, gpu_v4, NX, NY, Ein[0], Ein[1], E0);
        cudaStatus = cudaDeviceSynchronize();
        checkError(cudaStatus);

        /* Stages 3 and 4 */
        connectKernel << <numBlocks, numThreads >> > (gpu_v1, gpu_v2, gpu_v3, gpu_v4, gpu_out, NX, NY, n, Eout[0], Eout[1], rXmin, rXmax, rYmin, rYmax);
        cudaStatus = cudaDeviceSynchronize();
        checkError(cudaStatus);

        /* Debugging */
        if (n % 100 == 0) std::cout << n << std::endl;

    } // End of loop

    /* Stage 4: Output */
    cudaStatus = cudaMemcpy(h_out, gpu_out, (NT * sizeof(double)), cudaMemcpyDeviceToHost); // Memory Copy back to CPU
    checkError(cudaStatus);

    // Output timing and voltage at Eout point
    for (int i = 0; i < NT; ++i) {
        output << i * dt << "," << h_out[i] << std::endl; // Writes to file in comma delimited format
    }

    // Free allocated memory from GPU
    cudaFree(gpu_v1);
    cudaFree(gpu_v2);
    cudaFree(gpu_v3);
    cudaFree(gpu_v4);
    cudaFree(gpu_out);

    // Tidying up
    output.close();
    std::cout << "Done: " << ((std::clock() - start) / (double)CLOCKS_PER_SEC) << std::endl;
    std::cin.get();

} // end main

// EOF
