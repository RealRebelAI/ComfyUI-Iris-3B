"""Torchrun world topology: read it, prove it is one world, print it.

A hand-rolled ``torchrun --nproc_per_node=8`` started on two machines produces
two *unrelated* process groups, each with ranks 0..7, each convinced its world
size is 8. Both halves then walk the same data range and both elect a rank-0
checkpoint writer, so the run trains duplicate samples and the checkpoint is
whichever half wrote last. Nothing raises. ``assert_single_world`` exists to
turn that into a startup failure.
"""

import os
import socket
from dataclasses import dataclass

# Addresses a rendezvous can only reach within its own machine, so every node
# that uses one forms a private world.
_LOOPBACK = {"localhost", "localhost.localdomain", "127.0.0.1", "::1", "0.0.0.0", ""}


@dataclass(frozen=True)
class World:
    """Process placement as the launcher declared it, read once at startup."""

    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    node_rank: int
    num_nodes: int
    hostname: str
    master_addr: str
    master_port: int


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def describe_world() -> World:
    """Read the torchrun environment, defaulting to a single local process.

    ``num_nodes`` comes from torchrun's own ``GROUP_WORLD_SIZE`` and falls back
    to the ratio of world to local world size. ``NNODES`` is deliberately not
    consulted here: it is the operator's *declaration*, and checking one against
    the other is the whole point of ``assert_single_world``.
    """
    world_size = _env_int("WORLD_SIZE", 1)
    local_world_size = _env_int("LOCAL_WORLD_SIZE", 1)
    num_nodes = _env_int("GROUP_WORLD_SIZE", max(world_size // max(local_world_size, 1), 1))
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    return World(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
        node_rank=_env_int("GROUP_RANK", rank // max(local_world_size, 1)),
        num_nodes=num_nodes,
        hostname=socket.gethostname(),
        master_addr=os.environ.get("MASTER_ADDR", ""),
        master_port=_env_int("MASTER_PORT", 29500),
    )


def assert_single_world(world: World, expected_nodes: int | None = None) -> None:
    """Raise ValueError unless every rank of every node is in one process group.

    ``expected_nodes`` is the operator's declaration (``--expected-nodes``, or
    ``NNODES`` from the launcher). Passing it is what catches the failure that
    is otherwise undetectable from inside a process: a world that is internally
    consistent but is only half of the intended job.
    """
    if expected_nodes is None:
        declared = _env_int("NNODES", 0)
        expected_nodes = declared if declared > 0 else None

    if world.world_size != world.num_nodes * world.local_world_size:
        raise ValueError(
            f"split world: WORLD_SIZE={world.world_size} but "
            f"GROUP_WORLD_SIZE={world.num_nodes} x LOCAL_WORLD_SIZE={world.local_world_size} "
            f"= {world.num_nodes * world.local_world_size}. This rank can only see part of the "
            "job; the missing ranks form their own group with their own rank 0, so both halves "
            "read the same data range and both write checkpoints. Launch every node with "
            "scripts/launch_multinode.sh (--nnodes/--node-rank/--rdzv-endpoint)."
        )

    if world.num_nodes > 1 and world.master_addr in _LOOPBACK:
        raise ValueError(
            f"MASTER_ADDR={world.master_addr!r} is loopback but the job declares "
            f"{world.num_nodes} nodes. A loopback rendezvous is unreachable from any other "
            "machine, so each node forms its own world. Set MASTER_ADDR (or RDZV_ENDPOINT) to "
            "the routable address of node 0."
        )

    if expected_nodes is not None and expected_nodes != world.num_nodes:
        raise ValueError(
            f"node count mismatch: {expected_nodes} node(s) expected, {world.num_nodes} "
            f"observed (GROUP_WORLD_SIZE, WORLD_SIZE={world.world_size}, "
            f"LOCAL_WORLD_SIZE={world.local_world_size}). Either the launcher was started on "
            "the wrong number of hosts, or a host failed rendezvous and this world is only "
            "part of the job."
        )

    expected_rank = world.node_rank * world.local_world_size + world.local_rank
    if expected_rank != world.rank:
        raise ValueError(
            f"rank numbering is node-local: RANK={world.rank} but GROUP_RANK="
            f"{world.node_rank} x LOCAL_WORLD_SIZE={world.local_world_size} + LOCAL_RANK="
            f"{world.local_rank} = {expected_rank}. Ranks must be globally unique; duplicated "
            "ranks collide on the data range and the checkpoint writer."
        )


def log_world(world: World) -> str:
    """Print one line from every rank so a split launch is visible immediately."""
    line = (
        f"[world] host={world.hostname} rank={world.rank}/{world.world_size} "
        f"local_rank={world.local_rank}/{world.local_world_size} "
        f"node={world.node_rank}/{world.num_nodes} "
        f"master={world.master_addr}:{world.master_port}"
    )
    print(line, flush=True)
    return line
