#include "router_common.h"

namespace py = pybind11;

namespace moe_router {

std::tuple<Tensor, Tensor, Tensor> fused_topk_with_score_function_fwd(
    Tensor logits, int topk, bool use_pre_softmax, std::optional<int> num_groups,
    std::optional<int> group_topk, std::optional<double> scaling_factor,
    const std::string& score_function, std::optional<Tensor> expert_bias) {
  TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
  TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
  TORCH_CHECK(topk > 0, "topk must be greater than 0");
  const int num_tokens = logits.size(0);
  const int num_experts = logits.size(1);
  TORCH_CHECK(topk <= num_experts, "topk must be <= num_experts");

  if (expert_bias.has_value()) {
    TORCH_CHECK(expert_bias->is_cuda(), "expert_bias must be a CUDA tensor");
    TORCH_CHECK(expert_bias->dim() == 1 && expert_bias->size(0) == num_experts,
                "expert_bias must have shape [num_experts]");
    TORCH_CHECK(expert_bias->scalar_type() == torch::kFloat,
                "expert_bias must be float32");
    TORCH_CHECK(score_function == "sigmoid" || score_function == "sqrtsoftplus",
                "expert_bias is only supported for sigmoid and sqrtsoftplus");
  }

  int num_groups_value = num_groups.has_value() ? *num_groups : -1;
  int group_topk_value = group_topk.has_value() ? *group_topk : -1;
  if (group_topk_value > 0) {
    TORCH_CHECK(num_groups_value > 0, "num_groups must be set when group_topk is set");
    TORCH_CHECK(num_experts % num_groups_value == 0,
                "num_experts must be divisible by num_groups");
    TORCH_CHECK(topk % group_topk_value == 0, "topk must be divisible by group_topk");
  }

  float scaling_factor_value = scaling_factor.has_value() ? static_cast<float>(*scaling_factor) : 1.0f;
  int score_function_value = score_function_id(score_function);
  if (score_function_value != 1) {
    use_pre_softmax = false;
  }

  auto probs = torch::empty_like(logits);
  auto routing_map = torch::empty({num_tokens, num_experts}, logits.options().dtype(torch::kBool));
  auto intermediate_output =
      torch::empty({num_tokens, num_experts}, logits.options().dtype(torch::kFloat));

  auto expert_bias_tensor = expert_bias.has_value()
                                ? expert_bias->contiguous()
                                : torch::Tensor();
  auto logits_contig = logits.contiguous();
  launch_fused_topk_with_score_function_forward(
      logits_contig, num_tokens, num_experts, topk, use_pre_softmax, num_groups_value,
      group_topk_value, scaling_factor_value, score_function_value, expert_bias_tensor, probs,
      routing_map, intermediate_output);
  return {probs, routing_map, intermediate_output};
}

void fused_topk_with_score_function_bwd(int num_tokens, int num_experts, Tensor routing_map,
                                        Tensor intermediate_output, Tensor grad_probs,
                                        Tensor grad_logits, int topk, bool use_pre_softmax,
                                        std::optional<double> scaling_factor,
                                        const std::string& score_function) {
  TORCH_CHECK(routing_map.is_cuda() && intermediate_output.is_cuda() && grad_probs.is_cuda() &&
                  grad_logits.is_cuda(),
              "all tensors must be CUDA tensors");
  float scaling_factor_value = scaling_factor.has_value() ? static_cast<float>(*scaling_factor) : 1.0f;
  launch_fused_topk_with_score_function_backward(
      routing_map.contiguous(), intermediate_output.contiguous(), grad_probs.contiguous(), num_tokens,
      num_experts, topk, use_pre_softmax, scaling_factor_value, score_function_id(score_function),
      grad_logits);
}

std::tuple<Tensor, Tensor, Tensor> fused_score_for_moe_aux_loss_fwd(
    Tensor logits, int topk, const std::string& score_function) {
  TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
  TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
  const int num_tokens = logits.size(0);
  const int num_experts = logits.size(1);
  TORCH_CHECK(topk > 0 && topk <= num_experts, "topk must be in (0, num_experts]");
  TORCH_CHECK(num_experts % 32 == 0,
              "fused_score_for_moe_aux_loss_fwd currently requires num_experts to be a multiple of 32");

  auto logits_contig = logits.contiguous();
  auto scores = torch::empty({num_tokens, num_experts}, logits.options().dtype(torch::kFloat));
  auto routing_map = torch::empty({num_tokens, num_experts}, logits.options().dtype(torch::kBool));
  auto intermediate_output =
      torch::empty({num_tokens, num_experts}, logits.options().dtype(torch::kFloat));
  launch_fused_score_for_moe_aux_loss_forward(logits_contig, num_tokens, num_experts, topk,
                                              score_function_id(score_function), scores,
                                              routing_map, intermediate_output);
  return {scores, routing_map, intermediate_output};
}

void fused_score_for_moe_aux_loss_bwd(int num_tokens, int num_experts,
                                      Tensor intermediate_output, Tensor grad_scores,
                                      Tensor grad_logits, int topk,
                                      const std::string& score_function) {
  TORCH_CHECK(intermediate_output.is_cuda() && grad_scores.is_cuda() && grad_logits.is_cuda(),
              "all tensors must be CUDA tensors");
  launch_fused_score_for_moe_aux_loss_backward(intermediate_output.contiguous(),
                                               grad_scores.contiguous(), num_tokens, num_experts,
                                               topk, score_function_id(score_function), grad_logits);
}

std::tuple<Tensor, Tensor> fused_moe_aux_loss_fwd(Tensor probs, Tensor tokens_per_expert,
                                                  int total_num_tokens, int num_experts,
                                                  int num_rows, int num_cols, int topk,
                                                  double coeff) {
  TORCH_CHECK(probs.is_cuda() && tokens_per_expert.is_cuda(), "tensors must be CUDA tensors");
  auto aux_loss = torch::empty({}, probs.options());
  auto const_buf = torch::empty({}, probs.options().dtype(torch::kFloat));
  launch_fused_moe_aux_loss_forward(probs.contiguous(), tokens_per_expert.contiguous(),
                                    total_num_tokens, num_experts, num_rows, num_cols, topk,
                                    static_cast<float>(coeff), aux_loss, const_buf);
  return {aux_loss, const_buf};
}

Tensor fused_moe_aux_loss_bwd(Tensor const_buf, Tensor tokens_per_expert, int num_rows,
                              int num_cols, Tensor grad_aux_loss) {
  TORCH_CHECK(const_buf.is_cuda() && tokens_per_expert.is_cuda() && grad_aux_loss.is_cuda(),
              "tensors must be CUDA tensors");
  auto grad_probs =
      torch::empty({num_rows, num_cols}, grad_aux_loss.options().device(grad_aux_loss.device()));
  launch_fused_moe_aux_loss_backward(const_buf.contiguous(), tokens_per_expert.contiguous(),
                                     num_rows, num_cols, grad_aux_loss.contiguous(), grad_probs);
  return grad_probs;
}

}  // namespace moe_router

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_topk_with_score_function_fwd", &moe_router::fused_topk_with_score_function_fwd,
        py::arg("logits"), py::arg("topk"), py::arg("use_pre_softmax"), py::arg("num_groups"),
        py::arg("group_topk"), py::arg("scaling_factor"), py::arg("score_function"),
        py::arg("expert_bias"));
  m.def("fused_topk_with_score_function_bwd", &moe_router::fused_topk_with_score_function_bwd,
        py::arg("num_tokens"), py::arg("num_experts"), py::arg("routing_map"),
        py::arg("intermediate_output"), py::arg("grad_probs"), py::arg("grad_logits"),
        py::arg("topk"), py::arg("use_pre_softmax"), py::arg("scaling_factor"),
        py::arg("score_function"));
  m.def("fused_score_for_moe_aux_loss_fwd", &moe_router::fused_score_for_moe_aux_loss_fwd,
        py::arg("logits"), py::arg("topk"), py::arg("score_function"));
  m.def("fused_score_for_moe_aux_loss_bwd", &moe_router::fused_score_for_moe_aux_loss_bwd,
        py::arg("num_tokens"), py::arg("num_experts"), py::arg("intermediate_output"),
        py::arg("grad_scores"), py::arg("grad_logits"), py::arg("topk"),
        py::arg("score_function"));
  m.def("fused_moe_aux_loss_fwd", &moe_router::fused_moe_aux_loss_fwd, py::arg("probs"),
        py::arg("tokens_per_expert"), py::arg("total_num_tokens"), py::arg("num_experts"),
        py::arg("num_rows"), py::arg("num_cols"), py::arg("topk"), py::arg("coeff"));
  m.def("fused_moe_aux_loss_bwd", &moe_router::fused_moe_aux_loss_bwd, py::arg("const_buf"),
        py::arg("tokens_per_expert"), py::arg("num_rows"), py::arg("num_cols"),
        py::arg("grad_aux_loss"));
}
