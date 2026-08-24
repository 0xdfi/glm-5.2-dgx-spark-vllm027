# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch


def limit_draft_tokens(
    draft_tokens: torch.Tensor,
    num_speculative_tokens: int,
    max_num_speculative_tokens: int,
) -> torch.Tensor:
    """Limit a speculator's output to the scheduler-selected depth.

    Ported from R9/r15 payload (`AcceptanceLengthController` /
    `num_spec_tokens_to_schedule` plumbing): 0.27.1's own dynamic
    speculative-tokens schedule truncates draft depth *inside* the
    speculator/model-runner call sites via the `num_speculative_tokens`
    kwarg added to `propose()`, but does not provide this standalone
    defensive-slice helper. Kept as a second line of defense (e.g. for
    speculator implementations that ignore the kwarg and always return a
    full-depth draft tensor) and for the structured-output draft-token
    validation path in `spec_decode/utils.py`/`model_runner.py`.
    """
    if draft_tokens.ndim != 2:
        raise RuntimeError(
            "Speculator returned unsupported draft shape "
            f"{tuple(draft_tokens.shape)}; expected a 2D tensor."
        )
    if not 1 <= num_speculative_tokens <= max_num_speculative_tokens:
        raise RuntimeError(
            "Scheduler selected an invalid speculative-token count "
            f"{num_speculative_tokens}; expected a value between 1 and "
            f"{max_num_speculative_tokens}."
        )
    if draft_tokens.shape[1] == 0:
        return draft_tokens
    if draft_tokens.shape[1] < num_speculative_tokens:
        raise RuntimeError(
            "Speculator returned too few draft tokens "
            f"({draft_tokens.shape[1]}); expected at least "
            f"{num_speculative_tokens}."
        )
    return draft_tokens[:, :num_speculative_tokens]


class DraftTokensHandler:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        if not input_batch.has_structured_output_reqs:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self.draft_tokens_np = None
            return

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            self.draft_tokens_np = async_copy_to_np(draft_tokens)
            # draft_tokens is a temporary allocation on the main stream and read here on
            # copy_stream; without record_stream, the caching allocator may reuse its
            # memory before the async copy executes.
            draft_tokens.record_stream(self.copy_stream)
            self.copy_event.record()

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self.draft_tokens_np is not None:
            self.copy_event.synchronize()
            draft_token_ids = self.draft_tokens_np.tolist()
        else:
            # This case only happens when async scheduling is disabled.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        return DraftTokenIds(self.req_ids, draft_token_ids)


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
