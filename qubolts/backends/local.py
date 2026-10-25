"""Backends that run on the local machine: exact, simulated annealing, tabu, descent.

These are the "quantum-inspired" workhorses: they solve the same QUBO a quantum
annealer would, on commodity CPUs, at zero marginal cost.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import dimod
import numpy as np

from .base import Backend


class ExactBackend(Backend):
    """Brute force over all 2^n assignments. Ground truth for small problems."""

    name = "exact"
    max_variables = 22

    def sample(self, bqm: dimod.BinaryQuadraticModel, **kwargs: Any) -> dimod.SampleSet:
        if bqm.num_variables > self.max_variables:
            raise ValueError(
                f"exact backend limited to {self.max_variables} variables, got {bqm.num_variables}"
            )
        # return the lowest-energy states so the resolver can repair several candidates
        return dimod.ExactSolver().sample(bqm).truncate(kwargs.get("num_reads", 20))


class SimulatedAnnealingBackend(Backend):
    """Classical simulated annealing, with bounded parallel independent reads."""

    name = "sa"
    max_variables = 20_000

    def sample(self, bqm: dimod.BinaryQuadraticModel, **kwargs: Any) -> dimod.SampleSet:
        from dwave.samplers import SimulatedAnnealingSampler

        params = {"num_reads": 100, "num_sweeps": 2000}
        params.update(kwargs)
        requested_workers = params.pop("num_workers", 0)
        num_reads = int(params["num_reads"])
        if requested_workers is None or requested_workers == 0:
            workers = min(4, os.cpu_count() or 1, num_reads)
            if bqm.num_interactions > 250_000:
                workers = 1
        else:
            workers = int(requested_workers)
            if workers < 1:
                raise ValueError("num_workers must be a positive integer or 0 for auto")
            workers = min(workers, num_reads)

        if workers > 1 and bqm.num_interactions > 1_000_000:
            raise ValueError(
                "parallel SA is limited to QUBOs with at most 1,000,000 interactions "
                "to avoid duplicating large sampler graphs in memory"
            )
        if workers == 1:
            return SimulatedAnnealingSampler().sample(bqm, **params)

        seed = params.pop("seed", None)
        seeds = np.random.SeedSequence(seed).spawn(workers)
        read_counts = [num_reads // workers] * workers
        for i in range(num_reads % workers):
            read_counts[i] += 1
        sampler = SimulatedAnnealingSampler()

        def sample_chunk(read_count: int, child_seed: np.random.SeedSequence) -> dimod.SampleSet:
            chunk_params = params.copy()
            chunk_params["num_reads"] = read_count
            chunk_params["seed"] = (
                int(child_seed.generate_state(1, dtype=np.uint32)[0]) & 0x7FFFFFFF
            ) or 1
            return sampler.sample(bqm, **chunk_params)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(sample_chunk, read_count, child_seed)
                for read_count, child_seed in zip(read_counts, seeds, strict=True)
            ]
            return dimod.concatenate([future.result() for future in futures])


class TorchAnnealingBackend(Backend):
    """Batched simulated annealing on CUDA or Apple Metal through optional PyTorch.

    Updates are parallel across variables and reads, so this is an approximate
    synchronous annealer rather than the serial Metropolis chain used by ``sa``.
    """

    name = "torch_sa"
    max_variables = 20_000

    @staticmethod
    def _torch():
        try:
            import torch
        except ImportError:
            return None
        return torch

    def available(self) -> bool:
        torch = self._torch()
        return torch is not None and (
            torch.cuda.is_available()
            or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        )

    def sample(self, bqm: dimod.BinaryQuadraticModel, **kwargs: Any) -> dimod.SampleSet:
        torch = self._torch()
        if torch is None:
            raise RuntimeError("torch_sa requires the optional 'torch' dependency")

        requested_device = kwargs.pop("device", "auto")
        if requested_device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                raise RuntimeError("torch_sa requires a CUDA or Apple Metal (MPS) device")
        else:
            device = torch.device(requested_device)

        num_reads = int(kwargs.pop("num_reads", 32))
        num_sweeps = int(kwargs.pop("num_sweeps", 100))
        seed = kwargs.pop("seed", None)
        beta_range = kwargs.pop("beta_range", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported torch_sa sampler parameters: {names}")
        if num_reads < 1 or num_sweeps < 1:
            raise ValueError("num_reads and num_sweeps must be positive")
        if bqm.num_variables > self.max_variables:
            raise ValueError(
                f"torch_sa limited to {self.max_variables} variables, got {bqm.num_variables}"
            )
        if bqm.vartype is not dimod.BINARY:
            raise ValueError("torch_sa currently supports binary-valued BQMs only")

        variables = list(bqm.variables)
        num_variables = len(variables)
        matrix_bytes = num_variables * num_variables * np.dtype(np.float32).itemsize
        if matrix_bytes > 2 * 1024**3:
            raise ValueError(
                f"dense torch_sa interaction matrix needs {matrix_bytes / 1024**3:.2f} GiB; "
                "limit is 2 GiB"
            )

        vectors = bqm.to_numpy_vectors(variable_order=variables)
        quadratic = vectors.quadratic
        rows = quadratic.row_indices.astype(np.intp, copy=False)
        cols = quadratic.col_indices.astype(np.intp, copy=False)
        biases = np.asarray(quadratic.biases, dtype=np.float32)
        weights = np.zeros((num_variables, num_variables), dtype=np.float32)
        if len(biases):
            weights[rows, cols] = biases
            weights[cols, rows] = biases
        linear = np.asarray(vectors.linear_biases, dtype=np.float32)
        max_bias = max(
            float(np.max(np.abs(linear), initial=0.0)),
            float(np.max(np.abs(biases), initial=0.0)),
            1e-12,
        )
        if beta_range is None:
            beta_min, beta_max = 0.1 / max_bias, 5.0 / max_bias
        else:
            beta_min, beta_max = beta_range
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(int(seed))

        interaction = torch.as_tensor(weights, device=device)
        field = torch.as_tensor(linear, device=device)
        state = torch.randint(
            0,
            2,
            (num_reads, num_variables),
            device=device,
            generator=generator,
            dtype=torch.int8,
        ).to(torch.float32)
        for beta in np.linspace(beta_min, beta_max, num_sweeps):
            local_field = state @ interaction + field
            delta = (1.0 - 2.0 * state) * local_field
            probability = torch.exp(torch.clamp(-beta * delta, max=0.0))
            flip = (
                torch.rand(state.shape, device=device, generator=generator, dtype=torch.float32)
                < probability
            )
            state = torch.where(flip, 1.0 - state, state)

        energies = 0.5 * ((state @ interaction) * state).sum(dim=1) + state @ field
        energies += float(vectors.offset)
        samples = state.to(torch.int8).cpu().numpy()
        return dimod.SampleSet.from_samples(
            (samples, variables),
            vartype=dimod.BINARY,
            energy=energies.cpu().numpy().astype(np.float64),
            info={"device": str(device), "algorithm": "batched-synchronous-sa"},
        )


class TabuBackend(Backend):
    """Multistart tabu search (dwave-samplers). Strong on dense QUBOs."""

    name = "tabu"
    max_variables = 20_000

    def sample(self, bqm: dimod.BinaryQuadraticModel, **kwargs: Any) -> dimod.SampleSet:
        from dwave.samplers import TabuSampler

        params = {"num_reads": 10, "timeout": 100}  # timeout in ms per read
        params.update(kwargs)
        return TabuSampler().sample(bqm, **params)


class SteepestDescentBackend(Backend):
    """Greedy local descent from random starts. Fast, weak; a sanity floor."""

    name = "descent"
    max_variables = 100_000

    def sample(self, bqm: dimod.BinaryQuadraticModel, **kwargs: Any) -> dimod.SampleSet:
        from dwave.samplers import SteepestDescentSolver

        params = {"num_reads": 100}
        params.update(kwargs)
        return SteepestDescentSolver().sample(bqm, **params)
