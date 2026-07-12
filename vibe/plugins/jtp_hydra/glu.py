from math import sqrt

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import dropout, silu, softplus

__all__ = ("GLU", "SwiGLU", "SpGLU")


class GLU(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dropout: float = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.dropout = dropout

        self.weight = Parameter(
            torch.empty(
                out_features * 2,
                in_features,
                device=device,
                dtype=dtype,
            )
        )
        setattr(self.weight, "rr_muon_chunks", 2)

        torch.nn.init.kaiming_uniform_(self.weight, a=sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.weight.T
        x, x2 = x.chunk(2, dim=-1)
        x = self._activation(x)
        x = x * x2
        del x2

        if self.training and self.dropout != 0.0:
            x = dropout(x, self.dropout)

        return x

    @staticmethod
    def _activation(x: Tensor) -> Tensor:
        raise NotImplementedError


class SwiGLU(GLU):
    def _activation(self, x: Tensor) -> Tensor:
        return silu(x)


class SpGLU(GLU):
    def _activation(self, x: Tensor) -> Tensor:
        return softplus(x)
