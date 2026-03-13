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

template <typename T>
__device__ inline void fast_topk_and_mask(T *scores, int data_size, int topk, int *topk_indices,
                                          T *topk_scores, int lane_id) {
  // from manbo(https://github.com/XiaomingFun233)
  // Bit i indicates whether the i-th local element (lane_id + i * warp_size) was selected.
  uint32_t local_mask = 0;

  for (int k = 0; k < topk; k++) {
    CompType local_max_val = -std::numeric_limits<CompType>::infinity();
    int local_max_idx = -1;

    // 1) Per-lane local max on unmasked elements.
    int bit_idx = 0;
    for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
      CompType cur_val = 0.0f;
      if constexpr (std::is_same_v<CompType, double>) {
        uint64_t mask = -(uint64_t)((local_mask >> bit_idx) & 1u);
        uint64_t x_bits = __double_as_longlong(static_cast<CompType>(scores[i]));
        uint64_t result_bits =
          (~mask & x_bits) | (mask & 0xFFF0000000000000ULL);
        cur_val = __longlong_as_double(result_bits);  
      } else {
        uint32_t full_mask = -(uint32_t)((local_mask >> bit_idx) & 1u);
        uint32_t x_bits = __float_as_uint(static_cast<CompType>(scores[i]));
        uint32_t result_bits =
            (~full_mask & x_bits) | (full_mask & 0xFF800000u);
        cur_val = __uint_as_float(result_bits);
      }
      if (cur_val > local_max_val) {
        local_max_val = cur_val;
        local_max_idx = i;
      }
      bit_idx++;
    }

    // 2) Warp reduction to find global max and index.
    CompType global_max_val = local_max_val;
    int global_max_idx = local_max_idx;
    for (int s = kThreadsPerWarp / 2; s > 0; s /= 2) {
      CompType shuffled_val = __shfl_down_sync(0xffffffff, global_max_val, s);
      int shuffled_idx = __shfl_down_sync(0xffffffff, global_max_idx, s);
      if (shuffled_val > global_max_val) {
        global_max_val = shuffled_val;
        global_max_idx = shuffled_idx;
      }
    }
    global_max_idx = __shfl_sync(0xffffffff, global_max_idx, 0);
    global_max_val = __shfl_sync(0xffffffff, global_max_val, 0);

    // 3) Write top-k result.
    if (lane_id == 0) {
      topk_indices[k] = global_max_idx;
      topk_scores[k] = static_cast<T>(global_max_val);
    }

    // 4) Mark selected element in owning lane's local mask.
    if (global_max_idx >= 0 && (global_max_idx % kThreadsPerWarp) == lane_id) {
      int local_bit_pos = global_max_idx / kThreadsPerWarp;
      if (local_bit_pos < 32) {
        local_mask |= (1u << local_bit_pos);
      }
    }
  }

  // Keep a single sync point so call-sites can safely consume topk_* from all lanes.
  __syncwarp();
}

__device__ inline void naive_topk_and_mask(CompType* scores, int data_size, int topk,
                                           int* topk_indices, CompType* topk_scores,
                                           int lane_id) { 
  // from manbo(https://github.com/XiaomingFun233)
  fast_topk_and_mask(scores, data_size, topk, topk_indices, topk_scores, lane_id);
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
