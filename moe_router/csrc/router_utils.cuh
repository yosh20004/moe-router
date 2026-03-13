#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <limits>

namespace moe_router {

using CompType = float;

constexpr size_t kThreadsPerWarp = 32;
constexpr int kThreadsPerBlock = 128;
constexpr float epsilon = 1e-20f;

template <typename T>
__device__ inline T max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
__device__ inline T sum(T a, T b) {
  return a + b;
}

enum ReduceFuncType {
  SUM = 0,
  MAX = 1,
};

template <typename T>
__device__ inline T warp_reduce_on_shmem(T* data_ptr, int data_size, ReduceFuncType type,
                                         int lane_id) {
  T (*reduce_func)(T, T);
  CompType default_val = 0.0f;
  if (type == ReduceFuncType::SUM) {
    reduce_func = sum;
    default_val = 0.0f;
  } else {
    reduce_func = max;
    default_val = -std::numeric_limits<CompType>::infinity();
  }

  CompType val = lane_id < data_size ? data_ptr[lane_id] : default_val;
  for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
    val = reduce_func(val, data_ptr[i]);
  }

  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 16));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 8));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 4));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 2));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 1));
  __syncwarp();
  return T(val);
}

template <typename T>
__device__ inline T masked_warp_reduce_on_shmem(T* data_ptr, bool* mask, int data_size,
                                                ReduceFuncType type, int lane_id) {
  T (*reduce_func)(T, T);
  CompType default_val = 0.0f;
  if (type == ReduceFuncType::SUM) {
    reduce_func = sum;
    default_val = 0.0f;
  } else {
    reduce_func = max;
    default_val = -std::numeric_limits<CompType>::infinity();
  }

  CompType val = lane_id < data_size && mask[lane_id] ? data_ptr[lane_id] : default_val;
  for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
    if (mask[i]) {
      val = reduce_func(val, data_ptr[i]);
    }
  }

  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 16));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 8));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 4));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 2));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 1));
  __syncwarp();
  return T(val);
}

__device__ inline void apply_sigmoid_on_float(float* scores, int data_size, int lane_id) {
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = 1.0f / (1.0f + expf(-scores[i]));
  }
}

__device__ inline void apply_sigmoid_bwd_on_float(float* grad, float* fwd_output, int data_size,
                                                  int lane_id) {
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    grad[i] = grad[i] * fwd_output[i] * (1.0f - fwd_output[i]);
  }
}

__device__ inline void apply_sqrtsoftplus_on_float(float* scores, int data_size, int lane_id) {
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    float x = scores[i];
    float softplus_val = x > 20.0f ? x : log1pf(expf(x));
    scores[i] = sqrtf(softplus_val);
  }
}

__device__ inline void apply_sqrtsoftplus_bwd_on_float(float* grad, float* fwd_output,
                                                       float* logits_buf, int data_size,
                                                       int lane_id) {
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    float x = logits_buf[i];
    float y = fwd_output[i];
    float dy_dx = x > 20.0f ? 1.0f / (2.0f * y + epsilon)
                            : (1.0f / (1.0f + expf(-x))) / (2.0f * y + epsilon);
    grad[i] = grad[i] * dy_dx;
  }
}

__device__ inline void apply_softmax_bwd_on_float(float* grad, float* fwd_output,
                                                  float* comp_buf, bool* mask, int data_size,
                                                  int lane_id) {
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    if (mask) {
      comp_buf[i] = mask[i] ? grad[i] * fwd_output[i] : 0.0f;
    } else {
      comp_buf[i] = grad[i] * fwd_output[i];
    }
  }
  __syncwarp();
  float sum_output_x_grad =
      warp_reduce_on_shmem(comp_buf, data_size, ReduceFuncType::SUM, lane_id);
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    if (mask) {
      grad[i] = mask[i] ? fwd_output[i] * (grad[i] - sum_output_x_grad) : 0.0f;
    } else {
      grad[i] = fwd_output[i] * (grad[i] - sum_output_x_grad);
    }
  }
}

__device__ inline void apply_softmax_on_float(float* scores, int data_size, int lane_id) {
  float max_val = warp_reduce_on_shmem(scores, data_size, ReduceFuncType::MAX, lane_id);
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = expf(scores[i] - max_val);
  }
  __syncwarp();
  float sum_val = warp_reduce_on_shmem(scores, data_size, ReduceFuncType::SUM, lane_id);
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = scores[i] / sum_val;
  }
  __syncwarp();
}

__device__ inline void naive_topk_and_mask(CompType* scores, int data_size, int topk,
                                           int* topk_indices, CompType* topk_scores,
                                           int lane_id) {
  for (int k = 0; k < topk; k++) {
    bool lane_masked = false;
    for (int m = 0; m < k; m++) {
      if (topk_indices[m] == lane_id) {
        lane_masked = true;
        break;
      }
    }
    CompType val = (lane_id < data_size && !lane_masked) ? scores[lane_id]
                                                         : -std::numeric_limits<CompType>::infinity();
    int index = lane_id < data_size ? lane_id : 0;
    for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
      bool cur_masked = false;
      for (int m = 0; m < k; m++) {
        if (topk_indices[m] == i) {
          cur_masked = true;
          break;
        }
      }
      CompType cur_val = cur_masked ? -std::numeric_limits<CompType>::infinity() : scores[i];
      if (cur_val > val) {
        val = cur_val;
        index = i;
      }
    }
    for (int s = 16; s > 0; s /= 2) {
      auto shuffled_val = __shfl_xor_sync(0xffffffff, val, s);
      auto shuffled_index = __shfl_xor_sync(0xffffffff, index, s);
      if (shuffled_val > val) {
        val = shuffled_val;
        index = shuffled_index;
      }
    }
    if (lane_id == 0) {
      topk_indices[k] = index;
      topk_scores[k] = val;
    }
    __syncwarp();
  }
}

template <typename scalar_t>
__device__ inline CompType load_to_comp(const scalar_t* ptr) {
  return static_cast<CompType>(*ptr);
}

template <typename scalar_t>
__device__ inline scalar_t store_from_comp(CompType v) {
  return static_cast<scalar_t>(v);
}

}  // namespace moe_router
