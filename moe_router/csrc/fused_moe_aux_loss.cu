#include "router_common.h"
#include "router_utils.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace moe_router {

template <typename DataType, typename IndexType>
__global__ void fused_moe_aux_loss_forward_kernel(const DataType* probs,
                                                  const IndexType* tokens_per_expert,
                                                  int total_num_tokens, int num_experts,
                                                  int num_rows, int num_cols, int topk, float coeff,
                                                  DataType* aux_loss, float* const_buf);

template <typename DataType, typename IndexType>
__global__ void fused_moe_aux_loss_backward_kernel(const float* const_buf,
                                                   const IndexType* tokens_per_expert,
                                                   int num_rows, int num_cols,
                                                   DataType* grad_aux_loss,
                                                   DataType* grad_probs);

template <typename DataType>
void dispatch_fused_moe_aux_loss_forward_index(const Tensor& probs, const Tensor& tokens_per_expert,
                                               int total_num_tokens, int num_experts, int num_rows,
                                               int num_cols, int topk, float coeff,
                                               Tensor& aux_loss, Tensor& const_buf,
                                               cudaStream_t stream, size_t smem_size) {
  switch (tokens_per_expert.scalar_type()) {
    case at::kInt:
      fused_moe_aux_loss_forward_kernel<DataType, int><<<1, 1024, smem_size, stream>>>(
          probs.data_ptr<DataType>(), tokens_per_expert.data_ptr<int>(), total_num_tokens,
          num_experts, num_rows, num_cols, topk, coeff, aux_loss.data_ptr<DataType>(),
          const_buf.data_ptr<float>());
      break;
    case at::kLong:
      fused_moe_aux_loss_forward_kernel<DataType, int64_t><<<1, 1024, smem_size, stream>>>(
          probs.data_ptr<DataType>(), tokens_per_expert.data_ptr<int64_t>(), total_num_tokens,
          num_experts, num_rows, num_cols, topk, coeff, aux_loss.data_ptr<DataType>(),
          const_buf.data_ptr<float>());
      break;
    default:
      TORCH_CHECK(false, "tokens_per_expert must be int32 or int64");
  }
}

template <typename DataType>
void dispatch_fused_moe_aux_loss_backward_index(const Tensor& const_buf,
                                                const Tensor& tokens_per_expert, int num_rows,
                                                int num_cols, const Tensor& grad_aux_loss,
                                                Tensor& grad_probs, cudaStream_t stream,
                                                int grid_size, int block_size) {
  switch (tokens_per_expert.scalar_type()) {
    case at::kInt:
      fused_moe_aux_loss_backward_kernel<DataType, int><<<grid_size, block_size, 0, stream>>>(
          const_buf.data_ptr<float>(), tokens_per_expert.data_ptr<int>(), num_rows, num_cols,
          const_cast<Tensor&>(grad_aux_loss).data_ptr<DataType>(), grad_probs.data_ptr<DataType>());
      break;
    case at::kLong:
      fused_moe_aux_loss_backward_kernel<DataType, int64_t><<<grid_size, block_size, 0, stream>>>(
          const_buf.data_ptr<float>(), tokens_per_expert.data_ptr<int64_t>(), num_rows, num_cols,
          const_cast<Tensor&>(grad_aux_loss).data_ptr<DataType>(), grad_probs.data_ptr<DataType>());
      break;
    default:
      TORCH_CHECK(false, "tokens_per_expert must be int32 or int64");
  }
}

template <typename DataType, typename IndexType>
__global__ void fused_moe_aux_loss_forward_kernel(const DataType* probs,
                                                  const IndexType* tokens_per_expert,
                                                  int total_num_tokens, int num_experts,
                                                  int num_rows, int num_cols, int topk, float coeff,
                                                  DataType* aux_loss, float* const_buf) {
  if (blockIdx.x > 0) {
    return;
  }
  int warp_num = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem_aux_loss[];
  CompType* aggregated_probs_per_expert = reinterpret_cast<CompType*>(shmem_aux_loss);

  for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
    aggregated_probs_per_expert[i] = CompType(0);
  }
  __syncthreads();

  for (int i = lane_id; i < num_cols; i += kThreadsPerWarp) {
    CompType tmp = CompType(0);
    for (int j = warp_id; j < num_rows; j += warp_num) {
      tmp += static_cast<CompType>(probs[j * num_cols + i]);
    }
    atomicAdd(&aggregated_probs_per_expert[i], tmp);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
    aggregated_probs_per_expert[i] *= static_cast<CompType>(tokens_per_expert[i]);
  }
  __syncthreads();

  if (warp_id == 0) {
    CompType intermediate_result =
        warp_reduce_on_shmem(aggregated_probs_per_expert, num_cols, ReduceFuncType::SUM, lane_id);
    __syncwarp();
    if (lane_id == 0) {
      float c_coeff = (num_experts * coeff) / topk / total_num_tokens / total_num_tokens;
      aux_loss[0] = static_cast<DataType>(intermediate_result * c_coeff);
      const_buf[0] = c_coeff;
    }
  }
}

template <typename DataType, typename IndexType>
__global__ void fused_moe_aux_loss_backward_kernel(const float* const_buf,
                                                   const IndexType* tokens_per_expert,
                                                   int num_rows, int num_cols,
                                                   DataType* grad_aux_loss,
                                                   DataType* grad_probs) {
  int global_warp_num = gridDim.x * blockDim.x / kThreadsPerWarp;
  int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  for (int i = lane_id; i < num_cols; i += kThreadsPerWarp) {
    float c_coeff = const_buf[0];
    CompType tokens_per_expert_i = static_cast<CompType>(tokens_per_expert[i]);
    CompType grad_aux_loss_value = static_cast<CompType>(grad_aux_loss[0]);
    for (int j = global_warp_id; j < num_rows; j += global_warp_num) {
      grad_probs[j * num_cols + i] =
          static_cast<DataType>(c_coeff * tokens_per_expert_i * grad_aux_loss_value);
    }
  }
}

void launch_fused_moe_aux_loss_forward(const Tensor& probs, const Tensor& tokens_per_expert,
                                       int total_num_tokens, int num_experts, int num_rows,
                                       int num_cols, int topk, float coeff, Tensor& aux_loss,
                                       Tensor& const_buf) {
  const c10::cuda::OptionalCUDAGuard device_guard(probs.device());
  size_t smem_size = sizeof(CompType) * num_cols;
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (probs.scalar_type()) {
    case at::kFloat:
      dispatch_fused_moe_aux_loss_forward_index<float>(
          probs, tokens_per_expert, total_num_tokens, num_experts, num_rows, num_cols, topk,
          coeff, aux_loss, const_buf, stream.stream(), smem_size);
      break;
    case at::kHalf:
      dispatch_fused_moe_aux_loss_forward_index<at::Half>(
          probs, tokens_per_expert, total_num_tokens, num_experts, num_rows, num_cols, topk,
          coeff, aux_loss, const_buf, stream.stream(), smem_size);
      break;
    case at::kBFloat16:
      dispatch_fused_moe_aux_loss_forward_index<at::BFloat16>(
          probs, tokens_per_expert, total_num_tokens, num_experts, num_rows, num_cols, topk,
          coeff, aux_loss, const_buf, stream.stream(), smem_size);
      break;
    default:
      TORCH_CHECK(false, "Unsupported probs dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_fused_moe_aux_loss_backward(const Tensor& const_buf,
                                        const Tensor& tokens_per_expert, int num_rows,
                                        int num_cols, const Tensor& grad_aux_loss,
                                        Tensor& grad_probs) {
  const c10::cuda::OptionalCUDAGuard device_guard(grad_probs.device());
  int block_size = 256;
  int grid_size = (num_rows + block_size - 1) / block_size;
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (grad_probs.scalar_type()) {
    case at::kFloat:
      dispatch_fused_moe_aux_loss_backward_index<float>(
          const_buf, tokens_per_expert, num_rows, num_cols, grad_aux_loss, grad_probs,
          stream.stream(), grid_size, block_size);
      break;
    case at::kHalf:
      dispatch_fused_moe_aux_loss_backward_index<at::Half>(
          const_buf, tokens_per_expert, num_rows, num_cols, grad_aux_loss, grad_probs,
          stream.stream(), grid_size, block_size);
      break;
    case at::kBFloat16:
      dispatch_fused_moe_aux_loss_backward_index<at::BFloat16>(
          const_buf, tokens_per_expert, num_rows, num_cols, grad_aux_loss, grad_probs,
          stream.stream(), grid_size, block_size);
      break;
    default:
      TORCH_CHECK(false, "Unsupported grad_probs dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace moe_router
