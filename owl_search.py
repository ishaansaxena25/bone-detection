"""
Owl Search Algorithm (OSA) for Hyperparameter Optimisation.

Reference:
    Jain, M., Singh, V., & Rani, A. (2019).
    "A novel nature-inspired algorithm for optimization:
     Squirrel search algorithm." Swarm and Evolutionary Computation.

OSA simulates the nocturnal hunting behavior of owls:
  - Owls emit sounds and listen to echoes to locate prey (exploitation).
  - They randomly explore new areas when prey is not found (exploration).
  - The best-position memory acts as a global attractor.

Hyper-parameters tuned
----------------------
  x[0] : learning rate         [1e-5,  1e-2]
  x[1] : dropout rate          [0.10,  0.60]
  x[2] : weight decay (L2)     [1e-6,  1e-3]
  x[3] : momentum (SGD) / β₁   [0.80,  0.99]
  x[4] : FC-units index        [0,     3]   → maps to config.FC_UNIT_CHOICES
"""

import math
import random
import copy
import time
from typing import Callable, List

import numpy as np

import config


# ─── Helper ───────────────────────────────────────────────────────────────────

def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _decode(position: List[float]) -> dict:
    """Map raw OSA position vector → human-readable hyper-parameter dict."""
    fc_idx = int(round(_clip(position[4],
                             config.OSA_LOWER_BOUNDS[4],
                             config.OSA_UPPER_BOUNDS[4])))
    fc_idx = min(fc_idx, len(config.FC_UNIT_CHOICES) - 1)
    return {
        "lr":           _clip(position[0], *zip(config.OSA_LOWER_BOUNDS,
                                                 config.OSA_UPPER_BOUNDS))[0],
        "dropout":      _clip(position[1], config.OSA_LOWER_BOUNDS[1],
                               config.OSA_UPPER_BOUNDS[1]),
        "weight_decay": _clip(position[2], config.OSA_LOWER_BOUNDS[2],
                               config.OSA_UPPER_BOUNDS[2]),
        "momentum":     _clip(position[3], config.OSA_LOWER_BOUNDS[3],
                               config.OSA_UPPER_BOUNDS[3]),
        "fc_units":     config.FC_UNIT_CHOICES[fc_idx],
    }


# ─── OSA core ─────────────────────────────────────────────────────────────────

class Owl:
    """A single candidate solution (owl) in the search space."""

    def __init__(self, dim: int, lower: List[float], upper: List[float]):
        self.position = [
            random.uniform(lower[i], upper[i]) for i in range(dim)
        ]
        self.best_position = copy.deepcopy(self.position)
        self.fitness       = float("-inf")
        self.best_fitness  = float("-inf")

    def hyperparams(self) -> dict:
        return _decode(self.position)


class OwlSearchAlgorithm:
    """
    Owl Search Algorithm optimiser.

    Parameters
    ----------
    fitness_fn   : callable(hyperparams: dict) → float  (higher = better)
    population   : number of owls
    max_iter     : maximum optimisation iterations
    dim          : search-space dimensionality
    lower_bounds : per-dimension lower bound
    upper_bounds : per-dimension upper bound
    verbose      : print progress each iteration
    """

    def __init__(
        self,
        fitness_fn:   Callable[[dict], float],
        population:   int        = config.OSA_POPULATION,
        max_iter:     int        = config.OSA_MAX_ITER,
        dim:          int        = config.OSA_DIM,
        lower_bounds: List[float] = config.OSA_LOWER_BOUNDS,
        upper_bounds: List[float] = config.OSA_UPPER_BOUNDS,
        verbose:      bool       = True,
    ):
        self.fitness_fn   = fitness_fn
        self.population   = population
        self.max_iter     = max_iter
        self.dim          = dim
        self.lower        = lower_bounds
        self.upper        = upper_bounds
        self.verbose      = verbose

        self.owls: List[Owl] = []
        self.global_best_position: List[float] = []
        self.global_best_fitness:  float       = float("-inf")
        self.history: List[dict]               = []   # per-iteration log

    # ── Initialisation ────────────────────────────────────────────────────
    def _initialise(self):
        self.owls = [Owl(self.dim, self.lower, self.upper)
                     for _ in range(self.population)]
        print(f"\n[OSA] Initialising {self.population} owls …")
        for i, owl in enumerate(self.owls):
            owl.fitness = self.fitness_fn(owl.hyperparams())
            owl.best_fitness  = owl.fitness
            owl.best_position = copy.deepcopy(owl.position)
            if owl.fitness > self.global_best_fitness:
                self.global_best_fitness  = owl.fitness
                self.global_best_position = copy.deepcopy(owl.position)
            if self.verbose:
                print(f"  Owl {i+1:02d} | fitness={owl.fitness:.4f} | "
                      f"hp={owl.hyperparams()}")

    # ── Sound-wave update (exploitation) ──────────────────────────────────
    def _sound_update(self, owl: Owl, t: int):
        """
        Owls emit ultrasonic pulses and adjust position based on echo
        (exploitation toward personal & global best).
        """
        w   = 0.9 - 0.5 * (t / self.max_iter)   # inertia decays over time
        c1  = 2.0 * random.random()               # cognitive coefficient
        c2  = 2.0 * random.random()               # social coefficient

        new_pos = []
        for d in range(self.dim):
            vel = (
                w  * (owl.position[d] - owl.best_position[d])
                + c1 * (owl.best_position[d]        - owl.position[d])
                + c2 * (self.global_best_position[d] - owl.position[d])
            )
            new_d = _clip(owl.position[d] + vel, self.lower[d], self.upper[d])
            new_pos.append(new_d)
        owl.position = new_pos

    # ── Random flight (exploration) ───────────────────────────────────────
    def _random_flight(self, owl: Owl, t: int):
        """
        When no prey is detected nearby, owls fly randomly to explore
        new regions (Lévy-flight inspired step).
        """
        step = 1.0 - (t / self.max_iter)   # shrinks as iterations progress
        for d in range(self.dim):
            span  = (self.upper[d] - self.lower[d]) * step
            delta = random.gauss(0, span * 0.3)
            owl.position[d] = _clip(owl.position[d] + delta,
                                    self.lower[d], self.upper[d])

    # ── Main optimisation loop ────────────────────────────────────────────
    def run(self) -> dict:
        """
        Execute the Owl Search Algorithm.

        Returns
        -------
        best_hyperparams : dict of best found hyper-parameters
        """
        self._initialise()
        exploration_threshold = 0.3   # probability of random exploration

        for t in range(1, self.max_iter + 1):
            t0 = time.time()
            for owl in self.owls:
                if random.random() < exploration_threshold:
                    self._random_flight(owl, t)
                else:
                    self._sound_update(owl, t)

                # Evaluate
                owl.fitness = self.fitness_fn(owl.hyperparams())

                # Update personal best
                if owl.fitness > owl.best_fitness:
                    owl.best_fitness  = owl.fitness
                    owl.best_position = copy.deepcopy(owl.position)

                # Update global best
                if owl.fitness > self.global_best_fitness:
                    self.global_best_fitness  = owl.fitness
                    self.global_best_position = copy.deepcopy(owl.position)

            elapsed = time.time() - t0
            log = {
                "iteration":   t,
                "best_fitness": self.global_best_fitness,
                "best_hp":     _decode(self.global_best_position),
                "elapsed_s":   elapsed,
            }
            self.history.append(log)

            if self.verbose:
                print(
                    f"[OSA] Iter {t:03d}/{self.max_iter} | "
                    f"best_fitness={self.global_best_fitness:.4f} | "
                    f"hp={log['best_hp']} | {elapsed:.1f}s"
                )

        best_hp = _decode(self.global_best_position)
        print(f"\n[OSA] Optimisation complete.")
        print(f"      Best fitness : {self.global_best_fitness:.4f}")
        print(f"      Best HP      : {best_hp}")
        return best_hp

    def convergence_curve(self) -> List[float]:
        return [h["best_fitness"] for h in self.history]
