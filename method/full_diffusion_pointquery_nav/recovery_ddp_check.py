"""Check the real six-rank no_sync scaling against an eight-batch reference."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils import atomic_json, restore_rng, rng_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    config = json.loads(parser.parse_args().config.read_text())
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 6
    torch.manual_seed(17)
    model = torch.nn.Linear(7, 3).to(device)
    reference = torch.nn.Linear(7, 3).to(device)
    reference.load_state_dict(model.state_dict())
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False)
    values = [(torch.randn(32, 7, generator=torch.Generator().manual_seed(100 + v)).to(device),
               torch.randn(32, 3, generator=torch.Generator().manual_seed(200 + v)).to(device)) for v in range(8)]
    expected = sum((reference(x) - y).square().mean() / 8. for x, y in values)
    expected.backward()
    virtuals = config["virtual_rank_groups"][rank]
    for i, v in enumerate(virtuals):
        sync = nullcontext() if i == len(virtuals) - 1 else ddp.no_sync()
        with sync:
            loss = (ddp(values[v][0]) - values[v][1]).square().mean() * world / 8.
            loss.backward()
    for actual, wanted in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, wanted.grad, atol=2e-6, rtol=2e-5)
    # Each logical dropout/noise stream must survive another stream's execution.
    generator = torch.Generator(device=device).manual_seed(901 + rank)
    snapshot = rng_state(generator)
    wanted = torch.randn(32, 16, 2, generator=generator, device=device)
    torch.randn(20, device=device)
    restore_rng(snapshot, generator)
    torch.testing.assert_close(torch.randn(32, 16, 2, generator=generator, device=device), wanted, atol=0, rtol=0)
    dist.barrier()
    if rank == 0:
        atomic_json(Path(config["output"]) / "ddp_recovery_check.json", {"passed": True,
                    "physical_ranks": world, "logical_batches": 8, "global_batch": 256,
                    "uneven_no_sync_gradient_matches_eight_batch_reference": True,
                    "rng_roundtrip": True})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
