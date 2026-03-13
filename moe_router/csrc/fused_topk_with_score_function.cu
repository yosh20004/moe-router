#include "router_common.h"
#include "router_utils.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace moe_router {

template <typename DataType>
__global__ void fused_topk_with_score_function_forward_kernel(
    const DataType* logits, int num_tokens, int num_experts, int topk, bool use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const float* expert_bias, DataType* probs, bool* routing_map, CompType* intermediate_output);

template <typename DataType>
__global__ void fused_topk_with_score_function_backward_kernel(
    const bool* routing_map, const CompType* intermediate_output, const DataType* grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function, DataType* grad_logits);

template <typename DataType>
void dispatch_fused_topk_with_score_function_forward(
    const Tensor& logits, int num_tokens, int num_experts, int topk, bool use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const float* expert_bias_ptr, Tensor& probs, Tensor& routing_map, Tensor& intermediate_output,
    size_t grid_size, size_t shared_memory_size, cudaStream_t stream) {
  fused_topk_with_score_function_forward_kernel<DataType>
      <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
          logits.data_ptr<DataType>(), num_tokens, num_experts, topk, use_pre_softmax, num_groups,
          group_topk, scaling_factor, score_function, expert_bias_ptr, probs.data_ptr<DataType>(),
          routing_map.data_ptr<bool>(), intermediate_output.data_ptr<float>());
}

template <typename DataType>
void dispatch_fused_topk_with_score_function_backward(
    const Tensor& routing_map, const Tensor& intermediate_output, const Tensor& grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function, Tensor& grad_logits, size_t grid_size, size_t shared_memory_size,
    cudaStream_t stream) {
  fused_topk_with_score_function_backward_kernel<DataType>
      <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
          routing_map.data_ptr<bool>(), intermediate_output.data_ptr<float>(),
          grad_probs.data_ptr<DataType>(), num_tokens, num_experts, topk, use_pre_softmax,
          scaling_factor, score_function, grad_logits.data_ptr<DataType>());
}

template <typename DataType>
__global__ void fused_topk_with_score_function_forward_kernel(
    const DataType* logits, int num_tokens, int num_experts, int topk, bool use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const float* expert_bias, DataType* probs, bool* routing_map, CompType* intermediate_output) {
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem[];
  CompType* scores_buf = reinterpret_cast<CompType*>(shmem);
  CompType* topk_scores_buf = scores_buf + num_experts * num_token_per_block;
  CompType* group_scores_buf = nullptr;
  CompType* masked_scores_buf = nullptr;
  int* topk_indices_buf = nullptr;
  if (group_topk > 0) {
    masked_scores_buf = topk_scores_buf + topk * num_token_per_block;
    group_scores_buf = masked_scores_buf + num_experts * num_token_per_block;
    topk_indices_buf = reinterpret_cast<int*>(group_scores_buf + num_groups * num_token_per_block);
  } else {
    topk_indices_buf = reinterpret_cast<int*>(topk_scores_buf + topk * num_token_per_block);
  }

  CompType* scores = scores_buf + warp_id * num_experts;
  CompType* topk_scores = topk_scores_buf + warp_id * topk;
  CompType* masked_scores = masked_scores_buf ? masked_scores_buf + warp_id * num_experts : nullptr;
  CompType* group_scores = group_scores_buf ? group_scores_buf + warp_id * num_groups : nullptr;
  int* topk_indices = topk_indices_buf + warp_id * topk;

  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    if (token_offset_cur_warp >= num_tokens) {
      break;
    }
    int pos_offset = token_offset_cur_warp * num_experts;
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      probs[pos_offset + i] = DataType(0);
      routing_map[pos_offset + i] = false;
      if (score_function == 1) {
        intermediate_output[pos_offset + i] = -std::numeric_limits<CompType>::infinity();
      }
      scores[i] = static_cast<CompType>(logits[pos_offset + i]);
      if (group_topk > 0) {
        masked_scores[i] = -std::numeric_limits<CompType>::infinity();
      }
    }
    __threadfence_block();
    __syncwarp();

    if (use_pre_softmax && score_function == 1) {
      apply_softmax_on_float(scores, num_experts, lane_id);
      __syncwarp();
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = scores[i];
      }
    } else if (score_function == 0) {
      apply_sigmoid_on_float(scores, num_experts, lane_id);
      __syncwarp();
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = scores[i];
      }
    } else if (score_function == 2) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = scores[i];
      }
      __syncwarp();
      apply_sqrtsoftplus_on_float(scores, num_experts, lane_id);
    }
    __syncwarp();

    if (expert_bias != nullptr && (score_function == 0 || score_function == 2)) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        scores[i] += expert_bias[i];
      }
      __syncwarp();
    }

    if (group_topk > 0) {
      int group_size = num_experts / num_groups;
      for (int i = 0; i < num_groups; i++) {
        naive_topk_and_mask(scores + i * group_size, group_size, topk / group_topk, topk_indices,
                            topk_scores, lane_id);
        __syncwarp();
        if (lane_id == 0) {
          CompType tmp = 0.0f;
          for (int j = 0; j < topk / group_topk; j++) {
            tmp += topk_scores[j];
          }
          group_scores[i] = tmp;
        }
        __syncwarp();
      }
      naive_topk_and_mask(group_scores, num_groups, group_topk, topk_indices, topk_scores, lane_id);
      __syncwarp();
      for (int i = 0; i < group_topk; i++) {
        int st = topk_indices[i] * group_size;
        int ed = st + group_size;
        for (int j = st + lane_id; j < ed; j += kThreadsPerWarp) {
          masked_scores[j] = scores[j];
        }
      }
      __syncwarp();
      naive_topk_and_mask(masked_scores, num_experts, topk, topk_indices, topk_scores, lane_id);
    } else {
      naive_topk_and_mask(scores, num_experts, topk, topk_indices, topk_scores, lane_id);
    }
    __syncwarp();

    if (expert_bias != nullptr && (score_function == 0 || score_function == 2)) {
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        topk_scores[i] -= expert_bias[topk_indices[i]];
      }
      __syncwarp();
    }

    if (!use_pre_softmax && score_function == 1) {
      apply_softmax_on_float(topk_scores, topk, lane_id);
      __syncwarp();
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + topk_indices[i]] = topk_scores[i];
      }
      __syncwarp();
    }

    if (score_function == 0 || score_function == 2) {
      if (topk > 1) {
        CompType sum_scores = warp_reduce_on_shmem(topk_scores, topk, ReduceFuncType::SUM, lane_id);
        for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
          topk_scores[i] = topk_scores[i] / (sum_scores + epsilon);
        }
      }
      __syncwarp();
    }

    for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
      routing_map[pos_offset + topk_indices[i]] = true;
      probs[pos_offset + topk_indices[i]] = static_cast<DataType>(scaling_factor * topk_scores[i]);
    }
    __threadfence_block();
    __syncwarp();
  }
}

template <typename DataType>
__global__ void fused_topk_with_score_function_backward_kernel(
    const bool* routing_map, const CompType* intermediate_output, const DataType* grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function, DataType* grad_logits) {
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem[];
  CompType* grad_probs_buf = reinterpret_cast<CompType*>(shmem);
  CompType* act_from_fwd_buf = grad_probs_buf + num_experts * num_token_per_block;
  CompType* comp_buf = act_from_fwd_buf + num_experts * num_token_per_block;
  bool* routing_map_buf = reinterpret_cast<bool*>(comp_buf + num_experts * num_token_per_block);
  CompType* local_grad = grad_probs_buf + warp_id * num_experts;
  CompType* local_act_from_fwd = act_from_fwd_buf + warp_id * num_experts;
  CompType* local_comp_buf = comp_buf + warp_id * num_experts;
  bool* local_routing_map = routing_map_buf + warp_id * num_experts;

  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    if (token_offset_cur_warp >= num_tokens) {
      break;
    }
    int pos_offset = token_offset_cur_warp * num_experts;
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      local_grad[i] = static_cast<CompType>(grad_probs[pos_offset + i]);
      local_act_from_fwd[i] = intermediate_output[pos_offset + i];
      local_routing_map[i] = routing_map[pos_offset + i];
    }
    __threadfence_block();
    __syncwarp();

    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      if (local_routing_map[i]) {
        local_grad[i] = local_grad[i] * scaling_factor;
      }
    }
    __syncwarp();

    if (score_function == 2) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_comp_buf[i] = local_act_from_fwd[i];
      }
      __syncwarp();
      apply_sqrtsoftplus_on_float(local_comp_buf, num_experts, lane_id);
      __syncwarp();
    }

    if (topk > 1 && (score_function == 0 || score_function == 2)) {
      CompType* act_output = score_function == 0 ? local_act_from_fwd : local_comp_buf;
      CompType sum_fwd_input = masked_warp_reduce_on_shmem(act_output, local_routing_map,
                                                           num_experts, ReduceFuncType::SUM,
                                                           lane_id);
      CompType local_sum_output_x_grad = 0.0f;
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        if (local_routing_map[i]) {
          local_sum_output_x_grad += local_grad[i] * act_output[i];
        }
      }
      for (int s = 16; s > 0; s /= 2) {
        local_sum_output_x_grad += __shfl_xor_sync(0xffffffff, local_sum_output_x_grad, s);
      }
      CompType sum_output_x_grad = local_sum_output_x_grad;
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        if (local_routing_map[i]) {
          local_grad[i] = local_grad[i] / (sum_fwd_input + epsilon) -
                          sum_output_x_grad /
                              ((sum_fwd_input + epsilon) * (sum_fwd_input + epsilon));
        } else {
          local_grad[i] = 0.0f;
        }
      }
      __syncwarp();
    }

    if (!use_pre_softmax && score_function == 1) {
      apply_softmax_bwd_on_float(local_grad, local_act_from_fwd, local_comp_buf, local_routing_map,
                                 num_experts, lane_id);
      __syncwarp();
    }

    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      if (!local_routing_map[i]) {
        local_grad[i] = 0.0f;
      }
    }
    __syncwarp();

    if (score_function == 1 && use_pre_softmax) {
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

void launch_fused_topk_with_score_function_forward(const Tensor& logits, int num_tokens,
                                                   int num_experts, int topk,
                                                   bool use_pre_softmax, int num_groups,
                                                   int group_topk, float scaling_factor,
                                                   int score_function, const Tensor& expert_bias,
                                                   Tensor& probs, Tensor& routing_map,
                                                   Tensor& intermediate_output) {
  const c10::cuda::OptionalCUDAGuard device_guard(logits.device());
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(CompType) +
                              topk * num_token_per_block * sizeof(CompType) +
                              topk * num_token_per_block * sizeof(int);
  if (group_topk > 0) {
    shared_memory_size += num_groups * num_token_per_block * sizeof(CompType);
    shared_memory_size += num_experts * num_token_per_block * sizeof(CompType);
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  const float* expert_bias_ptr = expert_bias.defined() ? expert_bias.data_ptr<float>() : nullptr;
  switch (logits.scalar_type()) {
    case at::kFloat:
      dispatch_fused_topk_with_score_function_forward<float>(
          logits, num_tokens, num_experts, topk, use_pre_softmax, num_groups, group_topk,
          scaling_factor, score_function, expert_bias_ptr, probs, routing_map, intermediate_output,
          grid_size, shared_memory_size, stream.stream());
      break;
    case at::kHalf:
      dispatch_fused_topk_with_score_function_forward<at::Half>(
          logits, num_tokens, num_experts, topk, use_pre_softmax, num_groups, group_topk,
          scaling_factor, score_function, expert_bias_ptr, probs, routing_map, intermediate_output,
          grid_size, shared_memory_size, stream.stream());
      break;
    case at::kBFloat16:
      dispatch_fused_topk_with_score_function_forward<at::BFloat16>(
          logits, num_tokens, num_experts, topk, use_pre_softmax, num_groups, group_topk,
          scaling_factor, score_function, expert_bias_ptr, probs, routing_map, intermediate_output,
          grid_size, shared_memory_size, stream.stream());
      break;
    default:
      TORCH_CHECK(false, "Unsupported logits dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_fused_topk_with_score_function_backward(const Tensor& routing_map,
                                                    const Tensor& intermediate_output,
                                                    const Tensor& grad_probs, int num_tokens,
                                                    int num_experts, int topk,
                                                    bool use_pre_softmax,
                                                    float scaling_factor, int score_function,
                                                    Tensor& grad_logits) {
  const c10::cuda::OptionalCUDAGuard device_guard(grad_probs.device());
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(CompType) +
                              num_experts * num_token_per_block * sizeof(CompType) +
                              num_experts * num_token_per_block * sizeof(CompType) +
                              num_experts * num_token_per_block * sizeof(bool);
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (grad_logits.scalar_type()) {
    case at::kFloat:
      dispatch_fused_topk_with_score_function_backward<float>(
          routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
          use_pre_softmax, scaling_factor, score_function, grad_logits, grid_size,
          shared_memory_size, stream.stream());
      break;
    case at::kHalf:
      dispatch_fused_topk_with_score_function_backward<at::Half>(
          routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
          use_pre_softmax, scaling_factor, score_function, grad_logits, grid_size,
          shared_memory_size, stream.stream());
      break;
    case at::kBFloat16:
      dispatch_fused_topk_with_score_function_backward<at::BFloat16>(
          routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
          use_pre_softmax, scaling_factor, score_function, grad_logits, grid_size,
          shared_memory_size, stream.stream());
      break;
    default:
      TORCH_CHECK(false, "Unsupported grad_logits dtype");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace moe_router
