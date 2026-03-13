#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

#include <map>
#include <optional>
#include <string>
#include <tuple>

namespace moe_router {

using at::Tensor;

inline int score_function_id(const std::string& score_function) {
  static const std::map<std::string, int> score_function_map = {
      {"sigmoid", 0}, {"softmax", 1}, {"sqrtsoftplus", 2}};
  auto it = score_function_map.find(score_function);
  TORCH_CHECK(it != score_function_map.end(),
              "score_function must be softmax, sigmoid or sqrtsoftplus");
  return it->second;
}

std::tuple<Tensor, Tensor, Tensor> fused_topk_with_score_function_fwd(
    Tensor logits, int topk, bool use_pre_softmax, std::optional<int> num_groups,
    std::optional<int> group_topk, std::optional<double> scaling_factor,
    const std::string& score_function, std::optional<Tensor> expert_bias);

void fused_topk_with_score_function_bwd(int num_tokens, int num_experts, Tensor routing_map,
                                        Tensor intermediate_output, Tensor grad_probs,
                                        Tensor grad_logits, int topk, bool use_pre_softmax,
                                        std::optional<double> scaling_factor,
                                        const std::string& score_function);

std::tuple<Tensor, Tensor, Tensor> fused_score_for_moe_aux_loss_fwd(
    Tensor logits, int topk, const std::string& score_function);

void fused_score_for_moe_aux_loss_bwd(int num_tokens, int num_experts,
                                      Tensor intermediate_output, Tensor grad_scores,
                                      Tensor grad_logits, int topk,
                                      const std::string& score_function);

std::tuple<Tensor, Tensor> fused_moe_aux_loss_fwd(Tensor probs, Tensor tokens_per_expert,
                                                  int total_num_tokens, int num_experts,
                                                  int num_rows, int num_cols, int topk,
                                                  double coeff);

Tensor fused_moe_aux_loss_bwd(Tensor const_buf, Tensor tokens_per_expert, int num_rows,
                              int num_cols, Tensor grad_aux_loss);

void launch_fused_topk_with_score_function_forward(const Tensor& logits, int num_tokens,
                                                   int num_experts, int topk,
                                                   bool use_pre_softmax, int num_groups,
                                                   int group_topk, float scaling_factor,
                                                   int score_function, const Tensor& expert_bias,
                                                   Tensor& probs, Tensor& routing_map,
                                                   Tensor& intermediate_output);

void launch_fused_topk_with_score_function_backward(const Tensor& routing_map,
                                                    const Tensor& intermediate_output,
                                                    const Tensor& grad_probs, int num_tokens,
                                                    int num_experts, int topk,
                                                    bool use_pre_softmax,
                                                    float scaling_factor, int score_function,
                                                    Tensor& grad_logits);

void launch_fused_score_for_moe_aux_loss_forward(const Tensor& logits, int num_tokens,
                                                 int num_experts, int topk, int score_function,
                                                 Tensor& scores, Tensor& routing_map,
                                                 Tensor& intermediate_output);

void launch_fused_score_for_moe_aux_loss_backward(const Tensor& intermediate_output,
                                                  const Tensor& grad_scores, int num_tokens,
                                                  int num_experts, int topk, int score_function,
                                                  Tensor& grad_logits);

void launch_fused_moe_aux_loss_forward(const Tensor& probs, const Tensor& tokens_per_expert,
                                       int total_num_tokens, int num_experts, int num_rows,
                                       int num_cols, int topk, float coeff, Tensor& aux_loss,
                                       Tensor& const_buf);

void launch_fused_moe_aux_loss_backward(const Tensor& const_buf,
                                        const Tensor& tokens_per_expert, int num_rows,
                                        int num_cols, const Tensor& grad_aux_loss,
                                        Tensor& grad_probs);

}  // namespace moe_router
