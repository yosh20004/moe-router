from copy import deepcopy
from typing import Optional

import pytest
import torch

from moe_router.router import (
    fused_compute_score_for_moe_aux_loss,
    fused_moe_aux_loss,
    fused_topk_with_score_function,
)


seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


def group_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_tokens: int,
    num_experts: int,
    num_groups: int,
    group_topk: int,
):
    group_scores = (
        scores.view(num_tokens, num_groups, -1)
        .topk(topk // group_topk, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
    )
    masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    probs, top_indices = torch.topk(masked_scores, k=topk, dim=-1)
    return probs, top_indices


def topk_score_function_pytorch(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[torch.Tensor] = None,
):
    num_tokens, num_experts = logits.shape

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if group_topk:
            assert num_groups is not None
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        return torch.topk(scores, k=topk, dim=1)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    elif score_function in ("sigmoid", "sqrtsoftplus"):
        if score_function == "sigmoid":
            scores = torch.sigmoid(logits.float())
        else:
            scores = torch.nn.functional.softplus(logits.float()).sqrt()
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(
                scores_for_routing, topk, num_groups, group_topk
            )
            scores = torch.gather(scores, dim=1, index=top_indices)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = (
            scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
        )
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    probs = probs.type_as(logits)
    topk_masked_gates = torch.zeros_like(logits).scatter(1, top_indices, probs)
    topk_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return topk_masked_gates, topk_map


def compute_scores_for_aux_loss_pytorch(
    logits: torch.Tensor, topk: int, score_function: str
):
    if score_function == "softmax":
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float())
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    elif score_function == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(logits.float()).sqrt()
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"Invalid score_function: {score_function}")
    _, top_indices = torch.topk(scores, k=topk, dim=1)
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return routing_map, scores


def _should_run_group_topk(
    num_experts: int, topk: int, group_topk: Optional[int]
) -> bool:
    return group_topk is None or (num_experts % 8 == 0 and topk % group_topk == 0)


def aux_loss_pytorch(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: int,
    topk: int,
    num_experts: int,
    moe_aux_loss_coeff: float,
):
    aggregated_probs_per_expert = probs.sum(dim=0)
    return torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )


def _make_logits(dtype, num_tokens, num_experts, score_function):
    if score_function in ("sigmoid", "sqrtsoftplus"):
        offset = (
            torch.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype, device="cuda")
            * 1e-4
        )
        logits = (
            torch.arange(
                -num_experts // 2, num_experts // 2, device="cuda", dtype=dtype
            )
            * 1e-2
        )
        return logits.unsqueeze(0).repeat(num_tokens, 1) + offset.unsqueeze(1)
    logits = (
        torch.arange(
            -num_tokens * num_experts // 2,
            num_tokens * num_experts // 2,
            device="cuda",
            dtype=dtype,
        )
        * 1e-4
    )
    return logits.view(num_tokens, num_experts)


def run_comparison(
    dtype,
    num_tokens,
    num_experts,
    topk,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    score_function,
    enable_bias,
):
    logits = _make_logits(dtype, num_tokens, num_experts, score_function)
    logits.requires_grad = True
    if enable_bias and score_function in ("sigmoid", "sqrtsoftplus"):
        expert_bias = (
            torch.arange(num_experts, device="cuda", dtype=torch.float32) * 0.1
        )
        expert_bias = torch.flip(expert_bias, dims=[0])
        expert_bias.requires_grad = False
    else:
        expert_bias = None

    logits_clone = deepcopy(logits)
    logits_clone.requires_grad = True
    expert_bias_clone = deepcopy(expert_bias) if expert_bias is not None else None

    probs, routing_map = topk_score_function_pytorch(
        logits=logits,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function=score_function,
        expert_bias=expert_bias,
    )
    probs_fused, routing_map_fused = fused_topk_with_score_function(
        logits=logits_clone,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function=score_function,
        expert_bias=expert_bias_clone,
    )

    torch.testing.assert_close(probs, probs_fused)
    torch.testing.assert_close(routing_map, routing_map_fused)
    torch.sum(probs).backward()
    torch.sum(probs_fused).backward()
    torch.testing.assert_close(logits.grad, logits_clone.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("num_tokens", [256, 512])
@pytest.mark.parametrize("num_experts", [32, 128])
@pytest.mark.parametrize("topk", [4, 8])
@pytest.mark.parametrize("group_topk", [None, 4])
@pytest.mark.parametrize("scaling_factor", [None, 1.2])
@pytest.mark.parametrize("enable_bias", [True, False])
def test_topk_sigmoid(
    dtype, num_tokens, num_experts, topk, group_topk, scaling_factor, enable_bias
):
    if not _should_run_group_topk(num_experts, topk, group_topk):
        pytest.skip("invalid grouped topk configuration")
    num_groups = 8 if group_topk else None
    run_comparison(
        dtype,
        num_tokens,
        num_experts,
        topk,
        False,
        num_groups,
        group_topk,
        scaling_factor,
        "sigmoid",
        enable_bias,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("num_tokens", [256, 512])
@pytest.mark.parametrize("num_experts", [32, 128])
@pytest.mark.parametrize("topk", [4, 8])
@pytest.mark.parametrize("group_topk", [None, 4])
@pytest.mark.parametrize("scaling_factor", [None, 1.2])
@pytest.mark.parametrize("enable_bias", [True, False])
def test_topk_sqrtsoftplus(
    dtype, num_tokens, num_experts, topk, group_topk, scaling_factor, enable_bias
):
    if not _should_run_group_topk(num_experts, topk, group_topk):
        pytest.skip("invalid grouped topk configuration")
    num_groups = 8 if group_topk else None
    run_comparison(
        dtype,
        num_tokens,
        num_experts,
        topk,
        False,
        num_groups,
        group_topk,
        scaling_factor,
        "sqrtsoftplus",
        enable_bias,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("num_tokens", [256, 1024])
@pytest.mark.parametrize("num_experts", [32, 128])
@pytest.mark.parametrize("topk", [4, 8])
@pytest.mark.parametrize("use_pre_softmax", [True, False])
@pytest.mark.parametrize("group_topk", [None, 4])
@pytest.mark.parametrize("scaling_factor", [None, 1.2])
def test_topk_softmax(
    dtype, num_tokens, num_experts, topk, use_pre_softmax, group_topk, scaling_factor
):
    if not _should_run_group_topk(num_experts, topk, group_topk):
        pytest.skip("invalid grouped topk configuration")
    num_groups = 8 if group_topk else None
    run_comparison(
        dtype,
        num_tokens,
        num_experts,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        "softmax",
        False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("num_tokens", [256, 1024])
@pytest.mark.parametrize("num_experts", [32, 128])
@pytest.mark.parametrize("topk", [1, 4, 8])
@pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
def test_fused_scores_for_aux_loss(
    dtype, num_tokens, num_experts, topk, score_function
):
    logits = _make_logits(dtype, num_tokens, num_experts, score_function)
    logits.requires_grad = True
    logits_clone = deepcopy(logits)
    logits_clone.requires_grad = True

    routing_map, scores = compute_scores_for_aux_loss_pytorch(
        logits, topk, score_function
    )
    routing_map_fused, scores_fused = fused_compute_score_for_moe_aux_loss(
        logits_clone, topk, score_function
    )

    torch.testing.assert_close(scores, scores_fused)
    torch.testing.assert_close(routing_map, routing_map_fused)
    torch.sum(scores).backward()
    torch.sum(scores_fused).backward()
    torch.testing.assert_close(logits.grad, logits_clone.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("num_tokens", [256, 1024])
@pytest.mark.parametrize("num_experts", [32, 128])
@pytest.mark.parametrize("topk", [4])
def test_fused_moe_aux_loss(dtype, num_tokens, num_experts, topk):
    offset = (
        torch.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype, device="cuda")
        * 1e-4
    )
    probs = (
        torch.arange(-num_experts // 2, num_experts // 2, device="cuda", dtype=dtype)
        * 1e-2
    )
    probs = probs.unsqueeze(0).repeat(num_tokens, 1) + offset.unsqueeze(1)
    probs.requires_grad = True
    tokens_per_expert = torch.randint(
        1, 1000, (num_experts,), device="cuda", dtype=torch.int32
    )
    coeff = 0.01

    probs_clone = deepcopy(probs)
    probs_clone.requires_grad = True
    aux_loss = aux_loss_pytorch(
        probs, tokens_per_expert, num_tokens, topk, num_experts, coeff
    )
    aux_loss_fused = fused_moe_aux_loss(
        probs_clone,
        tokens_per_expert,
        total_num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        coeff=coeff,
    )

    torch.testing.assert_close(aux_loss, aux_loss_fused)
    aux_loss.backward()
    aux_loss_fused.backward()
    torch.testing.assert_close(probs.grad, probs_clone.grad)
