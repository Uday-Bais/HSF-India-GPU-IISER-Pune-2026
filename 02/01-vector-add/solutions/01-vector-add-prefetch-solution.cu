#include <stdio.h>  // Provides printf for status and error messages.

__global__  // Marks this function as a GPU kernel callable from the CPU.
void initWith(float num, float *a, int N)  // Fills every element of array a with the same value.
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;  // Compute this thread's global starting index.
  int stride = blockDim.x * gridDim.x;  // Compute how far to jump to reach this thread's next element.

  for(int i = index; i < N; i += stride)  // Walk through the array in a grid-stride loop.
  {
    a[i] = num;  // Write the chosen value into the current element.
  }
}

__global__  // Marks this function as a GPU kernel callable from the CPU.
void addVectorsInto(float *result, float *a, float *b, int N)  // Adds arrays a and b into result.
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;  // Compute this thread's global starting index.
  int stride = blockDim.x * gridDim.x;  // Compute the spacing between this thread's iterations.

  for(int i = index; i < N; i += stride)  // Walk through the full vector using a grid-stride loop.
  {
    result[i] = a[i] + b[i];  // Store the element-wise sum in the output vector.
  }
}

cudaError_t prefetchToDevice(void *pointer, size_t size, int deviceId)  // Prefetches managed memory to the active GPU.
{
#if CUDART_VERSION >= 13000  // Newer CUDA versions require a cudaMemLocation argument.
  cudaMemLocation location = {};  // Start with a zero-initialized destination descriptor.
  location.type = cudaMemLocationTypeDevice;  // State that the prefetch destination is a GPU.
  location.id = deviceId;  // Select which GPU device should receive the prefetched memory.
  return cudaMemPrefetchAsync(pointer, size, location, 0, 0);  // Enqueue the prefetch using the modern API.
#else  // Older CUDA versions still accept the device id directly.
  return cudaMemPrefetchAsync(pointer, size, deviceId);  // Enqueue the prefetch using the legacy API.
#endif  // End of the CUDA-version-specific compatibility path.
}

void checkElementsAre(float target, float *vector, int N)  // Verifies that every element matches the expected value.
{
  for(int i = 0; i < N; i++)  // Check each element one by one on the CPU.
  {
    if(vector[i] != target)  // Detect the first incorrect result.
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);  // Report which element is wrong.
      exit(1);  // Stop the program immediately after a failed verification.
    }
  }
  printf("Success! All values calculated correctly.\n");  // Report success when all checks pass.
}

int main()  // Runs the complete vector initialization and addition example.
{
  int deviceId;  // Will hold the id of the currently selected CUDA device.
  int numberOfSMs;  // Will hold the number of streaming multiprocessors on that device.

  cudaGetDevice(&deviceId);  // Query which GPU device is currently active.
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);  // Query how many SMs the active GPU has.

  const int N = 2<<24;  // Choose a large vector length so the GPU has substantial work to do.
  size_t size = N * sizeof(float);  // Compute how many bytes are needed for one vector.

  float *a;  // Managed pointer for the first input vector.
  float *b;  // Managed pointer for the second input vector.
  float *c;  // Managed pointer for the output vector.

  cudaMallocManaged(&a, size);  // Allocate unified memory for the first input vector.
  cudaMallocManaged(&b, size);  // Allocate unified memory for the second input vector.
  cudaMallocManaged(&c, size);  // Allocate unified memory for the output vector.

  prefetchToDevice(a, size, deviceId);  // Move vector a toward the GPU before it is used there.
  prefetchToDevice(b, size, deviceId);  // Move vector b toward the GPU before it is used there.
  prefetchToDevice(c, size, deviceId);  // Move vector c toward the GPU before it is written there.

  size_t threadsPerBlock;  // Will store how many threads each block should launch.
  size_t numberOfBlocks;  // Will store how many blocks to launch across the grid.

  threadsPerBlock = 256;  // Use a common block size that maps well to CUDA hardware.
  numberOfBlocks = 32 * numberOfSMs;  // Launch enough blocks to keep all SMs busy.

  cudaError_t addVectorsErr;  // Will hold any launch error from the last kernel launch.
  cudaError_t asyncErr;  // Will hold any asynchronous execution error after synchronization.

  initWith<<<numberOfBlocks, threadsPerBlock>>>(3, a, N);  // Fill vector a with the value 3 on the GPU.
  initWith<<<numberOfBlocks, threadsPerBlock>>>(4, b, N);  // Fill vector b with the value 4 on the GPU.
  initWith<<<numberOfBlocks, threadsPerBlock>>>(0, c, N);  // Initialize vector c to 0 on the GPU.

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);  // Add a and b together and store the result in c.

  addVectorsErr = cudaGetLastError();  // Check whether launching the kernels caused an immediate error.
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));  // Print the launch error if one occurred.

  asyncErr = cudaDeviceSynchronize();  // Wait for all GPU work to finish and surface asynchronous errors.
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));  // Print the asynchronous error if one occurred.

  checkElementsAre(7, c, N);  // Confirm that every output element equals 3 + 4.

  cudaFree(a);  // Release the managed memory used by vector a.
  cudaFree(b);  // Release the managed memory used by vector b.
  cudaFree(c);  // Release the managed memory used by vector c.
}
