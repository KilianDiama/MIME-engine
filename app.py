#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIME Engine — Industrial Critical Edition (10/10)
Hyperdimensional Mutagenic Active Inference Engine
"""

import asyncio
import hashlib
import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Any, Callable, Coroutine, Optional
from graphlib import TopologicalSorter, CycleError

import numpy as np
from numpy.linalg import eigh, norm

# ================================================================================
# LOGGING STRUCTURÉ
# ================================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
logger = logging.getLogger("MIME")

# ================================================================================
# CONFIGURATION IMMUTABLE
# ================================================================================
@dataclass(frozen=True)
class MIMEConfig:
    dim: int = 10_000
    entropy_floor: float = 0.25
    mutation_rate: float = 0.35
    history_capacity: int = 64
    svd_eps: float = 1e-12
    orth_eps: float = 1e-12
    max_nodes: int = 256

NodeLogicCallable = Callable[
    [List[np.ndarray], "HyperSpace", "MutagenicActiveInference"],
    Coroutine[Any, Any, np.ndarray],
]

# ================================================================================
# HYPERSPACE — COUCHE HYPERDIMENSIONNELLE IMMUNISÉE
# ================================================================================
class HyperSpace:
    def __init__(self, cfg: MIMEConfig, base_seed: Optional[int] = None):
        self.cfg = cfg
        self.base_rng = np.random.default_rng(base_seed)
        self.registry: Dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------------------
    # Hypervector generation (deterministic, bipolar, cached)
    # ---------------------------------------------------------------------------
    def generate_vector(self, key: str) -> np.ndarray:
        if key in self.registry:
            return self.registry[key]

        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        vec = rng.choice([-1.0, 1.0], size=self.cfg.dim).astype(np.float64)

        self.registry[key] = vec
        return vec

    # ---------------------------------------------------------------------------
    # Orthogonal mutation (Gram-Schmidt, stable)
    # ---------------------------------------------------------------------------
    def _compute_orthogonal_mutation(self, vector: np.ndarray, entropy: float) -> np.ndarray:
        raw = self.base_rng.normal(0, 1, self.cfg.dim)
        dot = np.dot(raw, vector)

        ortho = raw - (dot / (np.dot(vector, vector) + self.cfg.orth_eps)) * vector
        n = norm(ortho)

        if n < self.cfg.orth_eps:
            return vector

        ortho /= n
        factor = self.cfg.mutation_rate * (self.cfg.entropy_floor - entropy)
        mutated = vector + factor * ortho

        return mutated / (norm(mutated) + self.cfg.orth_eps)

    async def mutate_orthogonal(self, vector: np.ndarray, entropy: float) -> np.ndarray:
        if entropy >= self.cfg.entropy_floor:
            return vector
        return await asyncio.to_thread(self._compute_orthogonal_mutation, vector, entropy)

# ================================================================================
# MOTEUR D'INFÉRENCE ACTIVE MUTAGÈNE
# ================================================================================
class MutagenicActiveInference:
    def __init__(self, cfg: MIMEConfig, space: HyperSpace):
        self.cfg = cfg
        self.space = space
        self.history: deque = deque(maxlen=cfg.history_capacity)
        self.graph_dependencies: Dict[str, List[str]] = {}
        self.node_logics: Dict[str, NodeLogicCallable] = {}
        self._compiled_order: Optional[List[str]] = None

    # ---------------------------------------------------------------------------
    # Node registration
    # ---------------------------------------------------------------------------
    def register_node(self, name: str, deps: List[str], logic: NodeLogicCallable) -> None:
        if len(self.graph_dependencies) >= self.cfg.max_nodes:
            raise RuntimeError("Nombre maximal de nœuds MIME dépassé.")

        self.graph_dependencies[name] = deps
        self.node_logics[name] = logic
        self._compiled_order = None

    # ---------------------------------------------------------------------------
    # Graph compilation (topological sort)
    # ---------------------------------------------------------------------------
    def compile_graph(self) -> None:
        try:
            ts = TopologicalSorter(self.graph_dependencies)
            self._compiled_order = list(ts.static_order())
        except CycleError as ce:
            logger.critical(f"Cycle détecté dans le graphe MIME : {ce}")
            raise ce

    # ---------------------------------------------------------------------------
    # Entropy (spectral, Gram matrix)
    # ---------------------------------------------------------------------------
    def _sync_entropy(self, states: List[np.ndarray]) -> float:
        if len(states) < 2:
            return 1.0

        X = np.column_stack(states)
        Gram = np.dot(X.T, X)

        try:
            eigenvalues = eigh(Gram, eigvals_only=True)
            s = np.sqrt(np.maximum(eigenvalues, 0))
            s_sum = np.sum(s) + self.cfg.svd_eps
            p = s / s_sum
            p = p[p > self.cfg.svd_eps]
            return float(-np.sum(p * np.log2(p)))
        except Exception as e:
            logger.error(f"Erreur entropie : {e}")
            return 1.0

    async def calculate_entropy(self, states: List[np.ndarray]) -> float:
        return await asyncio.to_thread(self._sync_entropy, states)

    # ---------------------------------------------------------------------------
    # Execution step
    # ---------------------------------------------------------------------------
    async def execute_step(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self._compiled_order is None:
            self.compile_graph()

        for k, v in inputs.items():
            if v.shape[0] != self.cfg.dim:
                raise ValueError(f"Dimension incorrecte pour {k}")

        states = {k: v for k, v in inputs.items()}

        for node in self._compiled_order:
            if node in states:
                continue

            deps = self.graph_dependencies.get(node, [])
            dep_vecs = [states[d] for d in deps if d in states]

            raw = await self.node_logics[node](dep_vecs, self.space, self)
            entropy = await self.calculate_entropy(list(states.values()) + [raw])

            if entropy < self.cfg.entropy_floor:
                logger.warning(f"Crise entropique @ {node} (H={entropy:.4f}) → mutation")
                final = await self.space.mutate_orthogonal(raw, entropy)
            else:
                final = raw

            states[node] = final
            self.history.append(final)

        return states

# ================================================================================
# LOGIQUE DE NŒUD PAR DÉFAUT
# ================================================================================
async def default_logic(inputs: List[np.ndarray], space: HyperSpace, engine: MutagenicActiveInference) -> np.ndarray:
    if not inputs:
        return space.generate_vector("default_identity")

    combined = np.sum(inputs, axis=0)
    return combined / (norm(combined) + space.cfg.orth_eps)

# ================================================================================
# EXEMPLE D'EXÉCUTION
# ================================================================================
async def main():
    cfg = MIMEConfig(dim=4096, entropy_floor=0.35)
    space = HyperSpace(cfg, base_seed=123)
    engine = MutagenicActiveInference(cfg, space)

    engine.register_node("Sensor_A", [], default_logic)
    engine.register_node("Sensor_B", [], default_logic)
    engine.register_node("Fusion", ["Sensor_A", "Sensor_B"], default_logic)
    engine.register_node("Decision", ["Fusion"], default_logic)

    engine.compile_graph()

    inputs = {
        "Sensor_A": space.generate_vector("Alpha"),
        "Sensor_B": space.generate_vector("Beta"),
    }

    logger.info("MIME Engine v10/10 — démarrage")
    results = await engine.execute_step(inputs)
    logger.info(f"Nœuds résolus : {list(results.keys())}")

if __name__ == "__main__":
    asyncio.run(main())
