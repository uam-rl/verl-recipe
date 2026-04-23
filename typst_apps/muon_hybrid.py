"""Hybrid optimizer for LoRA RL runs.

Uses official torch.optim.Muon for matrix parameters and AdamW for any
non-matrix trainables. This keeps the run robust if the trainable set contains
LoRA metadata, biases, or other 1D parameters.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer


class MuonWithAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        defaults = dict(lr=lr, weight_decay=weight_decay)
        param_groups = self._split_param_groups(params, defaults)
        if not param_groups:
            raise ValueError("MuonWithAdamW got an empty parameter list")

        super().__init__(param_groups, defaults)

        optim_kinds = [group.pop("_optim") for group in self.param_groups]
        self._muon_indices = [i for i, kind in enumerate(optim_kinds) if kind == "muon"]
        self._adamw_indices = [i for i, kind in enumerate(optim_kinds) if kind == "adamw"]

        muon_groups = [self.param_groups[i] for i in self._muon_indices]
        adamw_groups = [self.param_groups[i] for i in self._adamw_indices]
        self.muon = (
            torch.optim.Muon(
                muon_groups,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                **{k: v for k, v in kwargs.items() if k in {"ns_coefficients", "ns_steps", "adjust_lr_fn"}},
            )
            if muon_groups
            else None
        )
        self.adamw = torch.optim.AdamW(adamw_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps) if adamw_groups else None

    @staticmethod
    def _as_groups(params: Iterable[torch.Tensor] | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        params = list(params)
        if not params:
            return []
        if isinstance(params[0], dict):
            return [dict(group) for group in params]  # shallow copy
        return [{"params": params}]

    @classmethod
    def _split_param_groups(
        cls,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        defaults: dict[str, Any],
    ) -> list[dict[str, Any]]:
        split_groups: list[dict[str, Any]] = []
        for group in cls._as_groups(params):
            group_defaults = {**defaults, **{k: v for k, v in group.items() if k != "params"}}
            matrix_params = []
            other_params = []
            for param in group["params"]:
                if not getattr(param, "requires_grad", True):
                    continue
                if getattr(param, "ndim", 0) == 2:
                    matrix_params.append(param)
                else:
                    other_params.append(param)
            if matrix_params:
                split_groups.append({**group_defaults, "params": matrix_params, "_optim": "muon"})
            if other_params:
                split_groups.append({**group_defaults, "params": other_params, "_optim": "adamw"})
        return split_groups

    def _sync_inner_group_options(self) -> None:
        if self.muon is not None:
            for inner_group, outer_idx in zip(self.muon.param_groups, self._muon_indices, strict=True):
                inner_group.update({k: v for k, v in self.param_groups[outer_idx].items() if k != "params"})
        if self.adamw is not None:
            for inner_group, outer_idx in zip(self.adamw.param_groups, self._adamw_indices, strict=True):
                inner_group.update({k: v for k, v in self.param_groups[outer_idx].items() if k != "params"})

    def step(self, closure=None):  # noqa: ANN001
        self._sync_inner_group_options()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.muon is not None:
            self.muon.step()
        if self.adamw is not None:
            self.adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {
            "param_groups": self.param_groups,
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.muon is not None and state_dict.get("muon") is not None:
            self.muon.load_state_dict(state_dict["muon"])
        if self.adamw is not None and state_dict.get("adamw") is not None:
            self.adamw.load_state_dict(state_dict["adamw"])
