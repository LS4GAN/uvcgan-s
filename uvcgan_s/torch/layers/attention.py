import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch import Tensor

from einops import rearrange
from uvcgan_s.torch.select import extract_name_kwargs

def expand_heads(values, n_heads):
    return rearrange(
        values, 'N L (D_h n_heads) -> (N n_heads) L D_h', n_heads = n_heads
    )

def contract_heads(values, n_heads):
    return rearrange(
        values, '(N n_heads) L D_h -> N L (D_h n_heads)', n_heads = n_heads
    )

def return_result_and_atten_weights(
    result, A, need_weights, average_attn_weights, batch_first, n_heads
):
    # pylint: disable=too-many-arguments

    # result : (N, L, embed_dim)
    # A : ((N n_heads), L, S)

    if not batch_first:
        # result : (N, L, embed_dim)
        #       -> (L, N, embed_dim)
        result = result.swapaxes(0, 1)

    if not need_weights:
        return (result, None)

    # A : (n_heads, N, L, S)
    A = rearrange(
        A, '(N n_heads) L S -> N n_heads L S', n_heads = n_heads
    )

    if average_attn_weights:
        # A : (N, n_heads, L, S)
        #  -> (N, L, S)
        A = A.mean(dim = 1)

    return (result, A)

class QuadraticAttention(nn.Module):

    # This is Lipshits continuous when equal_kq

    def __init__(
        self, embed_dim, num_heads,
        bias        = False,
        add_bias_kv = False,
        kdim        = None,
        vdim        = None,
        batch_first = False,
        equal_kq    = False,
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        if kdim is None:
            kdim = embed_dim
        elif equal_kq:
            assert kdim == embed_dim

        if vdim is None:
            vdim = embed_dim

        self._batch_first = batch_first
        self._n_heads     = num_heads
        self._dh          = embed_dim // num_heads

        self.w_q = nn.Linear(embed_dim, embed_dim, bias = bias)

        if equal_kq:
            self.w_k = self.w_q
        else:
            self.w_k = nn.Linear(kdim, embed_dim, bias = add_bias_kv)

        self.w_v = nn.Linear(vdim,      embed_dim, bias = add_bias_kv)
        self.w_o = nn.Linear(embed_dim, embed_dim, bias = bias)

    def compute_attention_matrix(self, w_query, w_key):
        # w_query : (N, L, d)
        # w_key   : (N, S, d)

        # w_query : (N, L,    d)
        #        -> (N, L, 1, d)
        w_query = w_query.unsqueeze(dim = 2)

        # w_key   : (N, S,    d)
        #        -> (N, 1, S, d)
        w_key   = w_key.unsqueeze(dim = 1)

        # L : (N, L, S)
        L = -torch.norm(w_query - w_key, p = 2, dim = 3)
        L = L / math.sqrt(self._dh)

        # result : (N, L, S)
        return torch.softmax(L, dim = 2)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor,
        key_padding_mask     : Optional[Tensor] = None,
        need_weights         : bool             = True,
        attn_mask            : Optional[Tensor] = None,
        average_attn_weights : bool             = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # pylint: disable=too-many-arguments

        assert key_padding_mask is None, "key_padding_mask is not supported"
        assert attn_mask is None,        "attn_mask is not supported"

        if not self._batch_first:
            # (k, q, v) : (L, N, ...)
            query = query.swapaxes(0, 1)
            key   =   key.swapaxes(0, 1)
            value = value.swapaxes(0, 1)

        # (query, key, value) : (N, L, ...)

        # w_query : ((N n_heads), L, D_h)
        # w_key   : ((N n_heads), S, D_h)
        # w_value : ((N n_heads), S, D_h)
        w_query = expand_heads(self.w_q(query), self._n_heads)
        w_key   = expand_heads(self.w_k(key),   self._n_heads)
        w_value = expand_heads(self.w_v(key),   self._n_heads)

        # A : ((N n_heads), L, S)
        A = self.compute_attention_matrix(w_query, w_key)

        # w_output : ((N n_heads), L, D_h)
        w_output = torch.bmm(A, w_value)

        # output : (N, L, embed_dim)
        output = contract_heads(w_output, self._n_heads)

        # result : (N, L, embed_dim)
        result = self.w_o(output)

        return return_result_and_atten_weights(
            result, A, need_weights, average_attn_weights,
            self._batch_first, self._n_heads
        )

def select_attention(attention, **extra_kwargs):
    name, kwargs = extract_name_kwargs(attention)

    if name in [ 'default', 'standard', 'scalar', 'dot' ]:
        return nn.MultiheadAttention(**kwargs, **extra_kwargs)

    if name in [ 'quadratic', 'l2' ]:
        return QuadraticAttention(**kwargs, **extra_kwargs)

    raise ValueError(f"Unknown attention {name}")

