import importlib.util
import sys
from pathlib import Path
from typing import Optional

import torch


def _load_extension():
    try:
        from . import _C as extension

        return extension
    except ImportError:
        package_root = Path(__file__).resolve().parent.parent
        build_root = package_root / "build"
        pattern = "lib.*/moe_router/_C*.so"
        candidates = sorted(build_root.glob(pattern), reverse=True)
        if not candidates:
            raise

        extension_path = candidates[0]
        spec = importlib.util.spec_from_file_location("moe_router._C", extension_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


_C = _load_extension()


class FusedTopkScoreFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        topk: int,
        use_pre_softmax: bool,
        num_groups: Optional[int],
        group_topk: Optional[int],
        scaling_factor: Optional[float],
        score_function: str,
        expert_bias: Optional[torch.Tensor],
    ):
        tensor_shape = logits.shape
        logits_2d = logits.contiguous().view(-1, tensor_shape[-1])
        probs, routing_map, intermediate_output = _C.fused_topk_with_score_function_fwd(
            logits_2d,
            topk,
            use_pre_softmax,
            num_groups,
            group_topk,
            scaling_factor,
            score_function,
            expert_bias,
        )
        ctx.save_for_backward(routing_map, intermediate_output)
        ctx.num_tokens = logits_2d.size(0)
        ctx.num_experts = logits_2d.size(1)
        ctx.topk = topk
        ctx.use_pre_softmax = use_pre_softmax
        ctx.scaling_factor = scaling_factor
        ctx.score_function = score_function
        ctx.logits_dtype = logits.dtype
        return probs.view(tensor_shape), routing_map.view(tensor_shape)

    @staticmethod
    def backward(ctx, grad_probs, _grad_routing_map):
        routing_map, intermediate_output = ctx.saved_tensors
        tensor_shape = grad_probs.shape
        grad_probs_2d = grad_probs.contiguous().view(-1, tensor_shape[-1])
        grad_logits = torch.empty(
            (ctx.num_tokens, ctx.num_experts),
            dtype=ctx.logits_dtype,
            device=grad_probs.device,
        )
        _C.fused_topk_with_score_function_bwd(
            ctx.num_tokens,
            ctx.num_experts,
            routing_map,
            intermediate_output,
            grad_probs_2d,
            grad_logits,
            ctx.topk,
            ctx.use_pre_softmax,
            ctx.scaling_factor,
            ctx.score_function,
        )
        return grad_logits.view(tensor_shape), None, None, None, None, None, None, None


def fused_topk_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: Optional[int],
    group_topk: Optional[int],
    scaling_factor: Optional[float],
    score_function: str,
    expert_bias: Optional[torch.Tensor],
):
    if logits.dtype == torch.float64:
        raise ValueError("Current router kernel does not support float64.")
    return FusedTopkScoreFunction.apply(
        logits,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        expert_bias,
    )


class FusedComputeScoresForMoEAuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, topk: int, score_function: str):
        tensor_shape = logits.shape
        logits_2d = logits.contiguous().view(-1, tensor_shape[-1])
        scores, routing_map, intermediate_output = _C.fused_score_for_moe_aux_loss_fwd(
            logits_2d,
            topk,
            score_function,
        )
        ctx.save_for_backward(intermediate_output)
        ctx.topk = topk
        ctx.score_function = score_function
        ctx.num_tokens = logits_2d.size(0)
        ctx.num_experts = logits_2d.size(1)
        ctx.logits_dtype = logits.dtype
        return routing_map.view(tensor_shape), scores.view(tensor_shape)

    @staticmethod
    def backward(ctx, _grad_routing_map, grad_scores):
        (intermediate_output,) = ctx.saved_tensors
        tensor_shape = grad_scores.shape
        grad_scores_2d = grad_scores.contiguous().view(-1, tensor_shape[-1])
        grad_logits = torch.empty(
            (ctx.num_tokens, ctx.num_experts),
            dtype=ctx.logits_dtype,
            device=grad_scores.device,
        )
        _C.fused_score_for_moe_aux_loss_bwd(
            ctx.num_tokens,
            ctx.num_experts,
            intermediate_output,
            grad_scores_2d,
            grad_logits,
            ctx.topk,
            ctx.score_function,
        )
        return grad_logits.view(tensor_shape), None, None


def fused_compute_score_for_moe_aux_loss(
    logits: torch.Tensor,
    topk: int,
    score_function: str,
):
    return FusedComputeScoresForMoEAuxLoss.apply(logits, topk, score_function)


class FusedAuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        total_num_tokens: int,
        num_experts: int,
        topk: int,
        coeff: float,
    ):
        probs_2d = probs.contiguous().view(-1, probs.shape[-1])
        aux_loss, const_buf = _C.fused_moe_aux_loss_fwd(
            probs_2d,
            tokens_per_expert.contiguous(),
            total_num_tokens,
            num_experts,
            probs_2d.size(0),
            probs_2d.size(1),
            topk,
            coeff,
        )
        ctx.save_for_backward(const_buf, tokens_per_expert)
        ctx.num_rows = probs_2d.size(0)
        ctx.num_cols = probs_2d.size(1)
        return aux_loss

    @staticmethod
    def backward(ctx, grad_aux_loss):
        const_buf, tokens_per_expert = ctx.saved_tensors
        grad_probs = _C.fused_moe_aux_loss_bwd(
            const_buf,
            tokens_per_expert,
            ctx.num_rows,
            ctx.num_cols,
            grad_aux_loss.contiguous(),
        )
        return grad_probs, None, None, None, None, None


def fused_moe_aux_loss(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: int,
    num_experts: int,
    topk: int,
    coeff: float,
) -> torch.Tensor:
    return FusedAuxLoss.apply(
        probs, tokens_per_expert, total_num_tokens, num_experts, topk, coeff
    )
