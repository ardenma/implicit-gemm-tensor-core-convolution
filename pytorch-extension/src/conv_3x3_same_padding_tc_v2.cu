#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// For tensor cores
#include <mma.h>
using namespace nvcuda;

// Available sizes: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-type-sizes

// GPU configuration.
#define WARP_SIZE 32

// MMA matrix tile dimensions. see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-type-sizes
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WMMA_INPUT_TILE_SIZE (WMMA_M * WMMA_K)
#define WMMA_FILTER_TILE_SIZE (WMMA_K * WMMA_N)

// Convolution parameters
#define KERNEL_SIZE 3
#define UNROLLED_KERNEL_SIZE 9
#define STRIDE 1

// Implementation constants.
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

// #define DEBUG

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
__global__ void conv_cuda_tc_forward_kernel_v2(
    const torch::PackedTensorAccessor32<at::Half,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<at::Half,4,torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor32<at::Half,4,torch::RestrictPtrTraits> output,
    void *output_ptr) {


  // Input: NHWC
  // Weights: KCRS
  // Output: NPQK

  // A (row maj): NHW x RSC    activations
  // B (col maj): RSC x K      filters
  // C (row maj): NPQ x K

  // Conmvolution problem size
  const int N = input.size(0); const int H = input.size(1);
  const int W = input.size(2); const int C = input.size(3);
  const int K = weights.size(0);
  const int R = weights.size(2);
  const int S = weights.size(3);
  const int P = output.size(1);
  const int Q = output.size(2);

  // Implicit GEMM matrix size
  int GEMM_M = N * P * Q;
  int GEMM_N = K;
  int GEMM_K = C * R * S;
  int SLICE_SIZE = UNROLLED_KERNEL_SIZE * C;

  // Row strides in the implicit GEMM matrix
  const int N_stride = H * W;
  const int H_stride = W;
  const int W_stride = 1;

  // Tile using a 2D grid (over the output), each threadblock
  // is (128, 2) -> (4,2) = 8 warps -> 32x64 output
  const int globalWarpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  const int globalWarpN = (blockIdx.y * blockDim.y + threadIdx.y);
  const int blockWarpM = threadIdx.x / WARP_SIZE;
  const int blockWarpN = threadIdx.y;
  const int numWarpsM = blockDim.x / WARP_SIZE;
  // const int numWarpsN = blockDim.y;
  const int intraWarpThreadIdx = threadIdx.x % WARP_SIZE;  // Thread idx within warp (0 to 31)
  const int blockLinearWarpIdx = (blockWarpN * numWarpsM) + blockWarpM;  // Warp idx within a block (0 to WARPS_PER_BLOCK - 1)

  // Shared memory tiles, currently only holds enough data for
  // each warp to have its own tile for a single MMA op (8 * 16 * 16 elements)
  // conceptually a WARPS_PER_BLOCK x (WMMA_M * WMMA_K) matrix
  // __shared__ __half smem_output_tile[WARPS_PER_BLOCK * WMMA_M * WMMA_N];
  __shared__ __half smem_input_tile[WARPS_PER_BLOCK * WMMA_M * WMMA_K];
  __shared__ __half smem_weight_tile[WARPS_PER_BLOCK * WMMA_K * WMMA_N];

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> acc_frag;
  wmma::fill_fragment(acc_frag, __float2half(0.0f));

  // Each warp reuses the same 16x16 tiles for all its computations
  __half *input_tile_start = smem_input_tile + (blockLinearWarpIdx * WMMA_INPUT_TILE_SIZE);
  __half *weight_tile_start = smem_weight_tile + (blockLinearWarpIdx * WMMA_FILTER_TILE_SIZE);

  // Loop over the K-dimension
  #pragma unroll
  for (int i = 0; i < GEMM_K; i += WMMA_K) {
    int aRow = globalWarpM * WMMA_M;
    int aCol = i;
    int bRow = i;
    int bCol = globalWarpN * WMMA_N;

    // Load into smem...
    // Each warp should load the 16x16 tile it's responsible for
    // i.e. each thread needs to load 8 elements of input and 8 elements of weight

    /**************************** Loading Input Tile ************************************/
    #pragma unroll
    for (int j = intraWarpThreadIdx; j < WMMA_INPUT_TILE_SIZE; j += WARP_SIZE) {
      // Compute where in the slice we are starting, e.g. the following
      // depicts slices bounded by | | symbols, and the start and end
      // of one row of the 16x16 WMMA matrix
      // row 0: [ *  *  * | *  s  *  *  *  *  *  *  * | *  *  *  *  *  *  *  e  * |  *  *  * ]
      //                       0  1  2  3  4  5  6  7  8   9 10 11 12 13 14 15
      // row 1: [ *  *  * | *  s  *  *  *  *  *  *  * | *  *  *  *  *  *  *  e  * |  *  *  * ]
      //                      16 17 18 19 20 21 22 23 24  25 26 27 28 29 30 31
      // row 3: [ *  *  * | *  s  *  *  *  *  *  *  * | *  *  *  *  *  *  *  e  * |  *  *  * ]
      //                      33 34 35 36 37 38 39 40 41  42 43 44 45 46 47 48

      // Slices are always 9 * C elements wide so we can compute where inside a slice
      // we are and also which row the slice is in relative to the start of the WMMA matrix
      int relSliceRow = j / WMMA_K;          // Relative row (0 - 15)
      int absSliceRow = aRow + relSliceRow;  // Row of the matrix the slice is on

      // Start index within a slice (0 to 9*C-1) that a half warp (16 threads) is responsible for
      // int sliceStartIdx_old = (aCol + relSliceRow * GEMM_K) % SLICE_SIZE;
      int sliceStartIdx = aCol % SLICE_SIZE;

      // Actual index within a slice (0 to 9*C-1) that the thread is repsonsible for
      int mySliceIdx = (sliceStartIdx + (j % WMMA_K)) % SLICE_SIZE;

      // Given the row of the matrix that the slice is in, and the index of the thread
      // within a slice, want to compute what input element to load...
      // first compute coordinates in output space (center of the kernel in MxK matrix A)
      int n = absSliceRow / N_stride;
      int p = (absSliceRow % N_stride) / H_stride;
      int q = ((absSliceRow % N_stride) % H_stride) / W_stride;

      // TODO map slice indexes to locations inside input... LUT per row?
      int offsets[3] = {-1, 0, 1};

      // Computing the coordinates of the center of the slice / kernel
      // in MxK matrix A, and using this information to figure out where
      // current thread's slice element lies. We want the following mapping:
      // 0 -> (-1, -1), 1 -> (0, -1), 2 -> (1, -1)
      // 3 -> (-1, 0),  4 -> (0, 0),  5 -> (1, 0)
      // 6 -> (-1, 1),  7 -> (0, 1),  8 -> (1, 1) 
      int y = p + offsets[(mySliceIdx % 9) / 3];
      int x = q + offsets[mySliceIdx % 3];
      int c = mySliceIdx / UNROLLED_KERNEL_SIZE;

      // Perform inbounds check
      if (x >= 0 && x < W && y >= 0 && y < H) {
          input_tile_start[j] = *static_cast<__half *>((void *) &input[n][y][x][c]);  // bruh dangerous
      } else {
          input_tile_start[j] = __float2half(0.f);
      }
    }

    /**************************** Loading Weight Tile ************************************/
    #pragma unroll
    for (int j = intraWarpThreadIdx; j < WMMA_FILTER_TILE_SIZE; j += WARP_SIZE) {
      // Basically follows the same slice logic as for the input tile but slices
      // are columns now
      
      int relSliceRow = j / WMMA_K;          // Relative row (0 - 15)
      int absSliceRow = bRow + relSliceRow;  // Row of the matrix the slice is on

      // int relSliceCol = j / WMMA_N;          // Relative col (0 - 15)
      int absSliceCol = bCol + (j % 16);  // Row of the matrix the slice is on

      // Given the row of the matrix that the slice is in, and the index of the thread
      // within a slice, want to compute what weight element to load...
      int k = absSliceCol;
      int c = absSliceRow / UNROLLED_KERNEL_SIZE;
      int r = (absSliceRow % UNROLLED_KERNEL_SIZE) / 3;
      int s = absSliceRow % 3;

      // Load weight
      weight_tile_start[j] = *static_cast<__half *>((void *) &weights[k][c][r][s]);  // bruh dangerous
    }

    /**************************** Bounds Check + WMMA Op ********************************/
    if (aRow < GEMM_M && aCol < GEMM_K && bRow < GEMM_K && bCol < GEMM_N) {
        // Load the inputs, final arg is 0 because memory is laid out linearly
        wmma::load_matrix_sync(a_frag, input_tile_start, WMMA_K);
        wmma::load_matrix_sync(b_frag, weight_tile_start, WMMA_N);

        // Perform the matrix multiplication
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha
  int cCol = globalWarpN * WMMA_N;
  int cRow = globalWarpM * WMMA_M;

  if (cRow < GEMM_M && cCol < GEMM_N) {
    // Store the output
    wmma::store_matrix_sync(static_cast<__half *>(output_ptr) + cCol + cRow * GEMM_N, acc_frag, GEMM_N, wmma::mem_row_major);
  }
}


std::vector<torch::Tensor> conv_cuda_tc_forward_v2(
    torch::Tensor input,
    torch::Tensor weights) {

  // Sanity check
  const auto batch_size = input.size(0);
  
  // Constructing output tensor (same HW as input)
  auto options =
  torch::TensorOptions()
    .dtype(torch::kFloat16)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  torch::Tensor output = torch::zeros({input.size(0), input.size(1), input.size(2), weights.size(0)}, options);

  // Defning problem shape
  const int N = input.size(0);
  const int K = weights.size(0);
  const int P = output.size(1);
  const int Q = output.size(2);
  
  // Converting conv problem shape to GEMM
  int GEMM_M = N * P * Q;
  int GEMM_N = K;

  // GEMM size must be a multiple of the WMMA size
  assert(GEMM_M % WMMA_M == 0);
  assert(GEMM_N % WMMA_N == 0);
//   int GEMM_K = C * R * S;
  
  // Defining launch configuration
  dim3 gridDim;
  dim3 blockDim;

  // blockDim.x must be a multple of warpSize
  // 128x2 means we have 8 warps and a block computes a 32x64 output tile
  blockDim.x = 128;
  blockDim.y = 2;
  assert(blockDim.y * blockDim.x / WARP_SIZE == WARPS_PER_BLOCK);
 
  gridDim.x = (GEMM_M + (WMMA_M * blockDim.x / 32 - 1)) /
              (WMMA_M * blockDim.x / 32);
  gridDim.y = (GEMM_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
  
  #ifdef DEBUG
  freopen("log.txt", "w+", stdout);
  printf("N: %d, P: %d, K: %d, Q: %d\n", N, P, K, Q);
  printf("GEMM_M: %d, GEMM_N: %d, GEMM_K: %d\n", GEMM_M, GEMM_N, 9 * input.size(3));
  printf("Grid: (%d, %d)\n", gridDim.x, gridDim.y);
  printf("Block: (%d, %d)\n", blockDim.x, blockDim.y);
  printf("Starting...\n");
  #endif

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(torch::kFloat16, "conv_forward_tc_cuda_v2", ([&] {
    conv_cuda_tc_forward_kernel_v2<<<gridDim, blockDim>>>(
        input.packed_accessor32<at::Half,4,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<at::Half,4,torch::RestrictPtrTraits>(),
        output.packed_accessor32<at::Half,4,torch::RestrictPtrTraits>(),
        output.data_ptr()
        );
  }));

  #ifdef DEBUG
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  printf("End...\n");
  #endif

  return {output};
}