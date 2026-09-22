# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os
from pathlib import Path

import pytest
import torch
import torch.distributed

from megatron.core.utils import is_te_min_version
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield
    if torch.distributed.is_initialized():
        print("Waiting for destroy_process_group")
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


@pytest.fixture(scope="session")
def ultraep_group(cleanup):
    """Share UltraEP's runtime across test files and release it before WORLD cleanup."""
    from megatron.core.transformer.moe.replica_weight_transport import (
        finalize_replica_weight_transports,
    )

    pytest.importorskip("ultra_ep")
    if not torch.cuda.is_available():
        pytest.skip("UltraEP requires CUDA/NVLink.")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    # Both native test modules use one EP group and one shared runtime lifetime.
    yield torch.distributed.group.WORLD
    torch.cuda.synchronize()
    torch.distributed.barrier()
    finalize_replica_weight_transports()


@pytest.fixture(scope="function", autouse=True)
def set_env():
    """Configure TE env vars for MoE unit tests.

    ``NVTE_CUTEDSL_FUSED_GROUPED_MLP`` enables TE's cuDSL fused grouped MLP path.
    The kernel additionally requires SM100 (Blackwell), so on H100/A100 CI this
    is a no-op; setting it here means the kernel is picked up automatically when
    Blackwell hardware joins the unit-test matrix.
    """
    if is_te_min_version("1.3"):
        os.environ['NVTE_FLASH_ATTN'] = '0'
        os.environ['NVTE_FUSED_ATTN'] = '0'
    os.environ['NVTE_CUTEDSL_FUSED_GROUPED_MLP'] = '1'


@pytest.fixture(scope="session")
def tmp_path_dist_ckpt(tmp_path_factory) -> Path:
    """Common directory for saving the checkpoint.

    Can't use pytest `tmp_path_factory` directly because directory must be shared between processes.
    """

    tmp_dir = tmp_path_factory.mktemp('ignored', numbered=False)
    tmp_dir = tmp_dir.parent.parent / 'tmp_dist_ckpt'

    if Utils.rank == 0:
        with TempNamedDir(tmp_dir, sync=False):
            yield tmp_dir

    else:
        yield tmp_dir
