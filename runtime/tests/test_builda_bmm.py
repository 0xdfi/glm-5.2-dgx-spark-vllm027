# Copyright 2026 dfi / 0xdfi
# SPDX-License-Identifier: Apache-2.0
"""GPU numeric checks for Build A; collected only inside the built SM121 image."""

from __future__ import annotations

import os

import pytest


@pytest.mark.requires_sm121
@pytest.mark.skipif(os.environ.get("CHAMPION_SM121_TESTS") != "1", reason="requires built SM121 image")
@pytest.mark.parametrize("m", range(1, 9))
@pytest.mark.parametrize("shape", [((16, 128, 512), 512), ((16, 512, 128), 128)])
@pytest.mark.parametrize("strided", [False, True])
def test_builda_matches_torch_bmm_for_deployed_shapes(m, shape, strided):
    import torch
    import triton  # noqa: F401 - proves the deployed runtime is importable
    from vllm.model_executor.layers.attention.builda_bmm_v0 import builda_bmm

    k, n = shape[0][1], shape[0][2]
    a = torch.randn((16, m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((16, k, n), device="cuda", dtype=torch.bfloat16)
    if strided:
        a = torch.randn((m, 16, k), device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    out = torch.empty((16, m, n), device="cuda", dtype=torch.bfloat16)
    if strided:
        out = torch.empty((m, 16, n), device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    expected = torch.bmm(a.float(), b.float()).to(torch.bfloat16)
    actual = builda_bmm(a, b, out)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.requires_sm121
@pytest.mark.skipif(os.environ.get("CHAMPION_SM121_TESTS") != "1", reason="requires built SM121 image")
def test_builda_falls_back_outside_shape_dtype_and_dimension_envelope(monkeypatch):
    import torch
    import vllm.model_executor.layers.attention.builda_bmm_v0 as builda

    calls = []
    original = torch.bmm
    monkeypatch.setattr(torch, "bmm", lambda a, b, out=None: calls.append((a.shape, a.dtype)) or original(a, b, out=out))
    for m, dtype in ((9, torch.bfloat16), (1, torch.float32)):
        a = torch.randn((16, m, 128), device="cuda", dtype=dtype)
        b = torch.randn((16, 128, 512), device="cuda", dtype=dtype)
        builda.builda_bmm(a, b, torch.empty((16, m, 512), device="cuda", dtype=dtype))
    assert len(calls) == 2
