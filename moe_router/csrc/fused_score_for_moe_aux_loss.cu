#include "router_common.h"
#include "router_utils.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace moe_router {

template <typename DataType>
__global__ void fused_score_for_moe_aux_loss_forward_kernel(const DataType* logits, int num_tokens,
                                                            int num_experts, int topk,
                                                            int score_function, float* scores,
                                                            bool* routing_map,
                                                            CompType* intermediate_output);

template <typename DataType>
__global__ void fused_score_for_moe_aux_loss_backward_kernel(const CompType* intermediate_output,
                                                             const float* grad_scores,
                                                             int num_tokens, int num_experts,
                                                             int topk, int score_function,
                                                             DataType* grad_logits);

template <typename DataType>
void dispatch_fused_score_for_moe_aux_loss_forward(const Tensor& logits, int num_tokens,
                                                   int num_experts, int topk, int score_function,
                                                   Tensor& scores, Tensor& routing_map,
                                                   Tensor& intermediate_output, size_t grid_size,
                                                   size_t shared_memory_size, cudaStream_t stream) {
  fused_score_for_moe_aux_loss_forward_kernel<DataType>
      <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
          logits.data_ptr<DataType>(), num_tokens, num_experts, topk, score_function,
          scores.data_ptr<float>(), routing_map.data_ptr<bool>(),
          intermediate_output.data_ptr<float>());
}

template <typename DataType>
void dispatch_fused_score_for_moe_aux_loss_backward(const Tensor& intermediate_output,
                                                    const Tensor& grad_scores, int num_tokens,
                                                    int num_experts, int topk, int score_function,
                                                    Tensor& grad_logits, size_t grid_size,
                                                    size_t shared_memory_size,
                                                    cudaStream_t stream) {
  fused_score_for_moe_aux_loss_backward_kernel<DataType>
      <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
          intermediate_output.data_ptr<float>(), grad_scores.data_ptr<float>(), num_tokens,
          num_experts, topk, score_function, grad_logits.data_ptr<DataType>());
}

template <typename DataType>
__global__ void fused_score_for_moe_aux_loss_forward_kernel(const DataType* logits, int num_tokens,
                                                            int num_experts, int topk,
                                                            int score_function, float* scores,
                                                            bool* routing_map,
                                                            CompType* intermediate_output) {
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem_scores_for_aux_loss[];
  CompType* logits_buf = reinterpret_cast<CompType*>(shmem_scores_for_aux_loss);
  CompType* topk_logits_buf = reinterpret_cast<CompType*>(logits_buf + num_experts * num_token_per_block);
  int* topk_indices_buf = reinterpret_cast<int*>(topk_logits_buf + topk * num_token_per_block);
  CompType* local_logits = logits_buf + warp_id * num_experts;
  CompType* topk_logits = topk_logits_buf + warp_id * topk;
  int* topk_indices = topk_indices_buf + warp_id * topk;

  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    if (token_offset_cur_warp >= num_tokens) {
      break;
    }
    int pos_offset = token_offset_cur_warp * num_experts;
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      routing_map[pos_offset + i] = false;
      if (score_function == 1) {
        intermediate_output[pos_offset + i] = -std::numeric_limits<CompType>::infinity();
      }
      local_logits[i] = static_cast<CompType>(logits[pos_offset + i]);
    }
    __threadfence_block();
    __syncwarp();

    if (score_function == 1) {
      apply_softmax_on_float(local_logits, num_experts, lane_id);
      __syncwarp();
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = local_logits[i];
      }
    } else if (score_function == 0) {
      apply_sigmoid_on_float(local_logits, num_experts, lane_id);
      __syncwarp();
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = local_logits[i];
      }
    } else if (score_function == 2) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = local_logits[i];
      }
      __syncwarp();
      apply_sqrtsoftplus_on_float(local_logits, num_experts, lane_id);
    }
    __syncwarp();

    if (score_function == 0 || score_function == 2) {
      auto sum_logits = warp_reduce_on_shmem(local_logits, num_experts, ReduceFuncType::SUM, lane_id);
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_logits[i] /= (sum_logits + epsilon);
      }
      __syncwarp();
    }

    naive_topk_and_mask(local_logits, num_experts, topk, topk_indices, topk_logits, lane_id);
    __syncwarp();

    for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
      routing_map[pos_offset + topk_indices[i]] = true;
    }
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      scores[pos_offset + i] = local_logits[i];
    }
    __threadfence_block();
    __syncwarp();
  }
}

template <typename DataType>
__global__ void fused_score_for_moe_aux_loss_backward_kernel(const CompType* intermediate_output,
                                                             const float* grad_scores,
                                                             int num_tokens, int num_experts,
                                                             int topk, int score_function,
                                                             DataType* grad_logits) {
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem[];
  CompType* grad_scores_buf = reinterpret_cast<CompType*>(shmem);
  CompType* act_from_fwd_buf = grad_scores_buf + num_experts * num_token_per_block;
  CompType* comp_buf = act_from_fwd_buf + num_experts * num_token_per_block;
  CompType* local_grad = grad_scores_buf + warp_id * num_experts;
  CompType* local_act_from_fwd = act_from_fwd_buf + warp_id * num_experts;
  CompType* local_comp_buf = comp_buf + warp_id * num_experts;

  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    if (token_offset_cur_warp >= num_tokens) {
      break;
    }
    int pos_offset = token_offset_cur_warp * num_experts;
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      local_grad[i] = grad_scores[pos_offset + i];
      local_act_from_fwd[i] = intermediate_output[pos_offset + i];
    }
    __threadfence_block();
    __syncwarp();

    if (score_function == 2) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_comp_buf[i] = local_act_from_fwd[i];
      }
      __syncwarp();
      apply_sqrtsoftplus_on_float(local_comp_buf, num_experts, lane_id);
      __syncwarp();
    }

    if (score_function == 0 || score_function == 2) {
      CompType* act_output = score_function == 0 ? local_act_from_fwd : local_comp_buf;
      auto sum_fwd_input = warp_reduce_on_shmem(act_output, num_experts, ReduceFuncType::SUM, lane_id);
      CompType local_sum_output_x_grad = 0.0f;
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_sum_output_x_grad += local_grad[i] * act_output[i];
      }
      for (int s = 16; s > 0; s /= 2) {
        local_sum_output_x_grad += __shfl_xor_sync(0xffffffff, local_sum_output_x_grad, s);
      }
      CompType sum_output_x_grad = local_sum_output_x_grad;
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_grad[i] = local_grad[i] / (sum_fwd_input + epsilon) -
                        sum_output_x_grad / ((sum_fwd_input + epsilon) * (sum_fwd_input + epsilon));
      }
      __syncwarp();
    }

    if (score_function == 1) {
      apply_softmax_bwd_on_float(local_grad, local_act_from_fwd, local_comp_buf, nullptr,
                                 num_experts, lane_id);
      __syncwarp();
    }
    if (score_function == 0) {
      apply_sigmoid_bwd_on_float(local_grad, local_act_from_fwd, num_experts, lane_id);
      __syncwarp();
    }
    if (score_function == 2) {
      apply_sqrtsoftplus_bwd_on_float(local_grad, local_comp_buf, local_act_from_fwd, num_experts,
                                      lane_id);
      __syncwarp();
    }
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      grad_logits[pos_offset + i] = static_cast<DataType>(local_grad[i]);
    }
    __syncwarp();
  }
}

void launch_fused_score_for_moe_aux_loss_forward(const Tensor& logits, int num_tokens,
                                                 int num_experts, int topk, int score_function,
                                                 Tensor& scores, Tensor& routing_map,
                                                 Tensor& intermediate_output) {
  const c10::cuda::OptionalCUDAGuard device_guard(logits.device());
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(CompType) +
                              topk * num_token_per_block * sizeof(CompType) +
                              topk * num_token_per_block * sizeof(int);
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (logits.scalar_type()) {
    case at::kFloat:
      dispatch_fused_score_for_moe_aux_loss_forward<float>(
          logits, num_tokens, num_experts, topk, score_function, scores, routing_map,
          intermediate_output, grid_size, shared_memory_size, stream.stream());
      break;
    case at::kHalf:
      dispatch_fused_score_for_moe_aux_loss_forward<at::Half>(
          logits, num_tokens, num_experts, topk, score_function, scores, routing_map,
          intermediate_output, grid_size, shared_memory_size, stream.stream());
      break;
    case at::kBFloat16:
      dispatch_fused_score_for_moe_aux_loss_forward<at::BFloat16>(
          logits, num_tokens, num_experts, topk, score_function, scores, routing_map,
          intermediate_output, grid_size, shared_memory_size, stream.stream());
      break;
    default:
      TORCH_CHECK(false, "Unsupported logits dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_fused_score_for_moe_aux_loss_backward(const Tensor& intermediate_output,
                                                  const Tensor& grad_scores, int num_tokens,
                                                  int num_experts, int topk, int score_function,
                                                  Tensor& grad_logits) {
  const c10::cuda::OptionalCUDAGuard device_guard(grad_logits.device());
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(CompType) +
                              num_experts * num_token_per_block * sizeof(CompType) +
                              num_experts * num_token_per_block * sizeof(CompType);
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (grad_logits.scalar_type()) {
    case at::kFloat:
      dispatch_fused_score_for_moe_aux_loss_backward<float>(
          intermediate_output, grad_scores, num_tokens, num_experts, topk, score_function,
          grad_logits, grid_size, shared_memory_size, stream.stream());
      break;
    case at::kHalf:
      dispatch_fused_score_for_moe_aux_loss_backward<at::Half>(
          intermediate_output, grad_scores, num_tokens, num_experts, topk, score_function,
          grad_logits, grid_size, shared_memory_size, stream.stream());
      break;
    case at::kBFloat16:
      dispatch_fused_score_for_moe_aux_loss_backward<at::BFloat16>(
          intermediate_output, grad_scores, num_tokens, num_experts, topk, score_function,
          grad_logits, grid_size, shared_memory_size, stream.stream());
      break;
    default:
      TORCH_CHECK(false, "Unsupported grad_logits dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace moe_router
