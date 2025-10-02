"""Probability distribution helpers for the process simulator."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Dict, Optional


@dataclass
class DistributionConfig:
    """User friendly specification of a probability distribution."""

    type: str
    parameters: Dict[str, float]

    def describe(self) -> str:
        items = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.type.title()}({items})"


class Distribution:
    """Base distribution interface."""

    def __init__(self, sampler: Callable[[random.Random], float], description: str) -> None:
        self._sampler = sampler
        self.description = description

    def sample(self, rng: random.Random) -> float:
        value = self._sampler(rng)
        return max(0.0, value)


class DistributionFactory:
    """Factory for constructing distributions from configuration dictionaries."""

    SUPPORTED_TYPES = {"uniform", "triangular", "normal", "lognormal", "exponential"}

    @staticmethod
    def from_config(config: DistributionConfig) -> Distribution:
        dist_type = config.type.lower().strip()
        params = config.parameters
        if dist_type == "uniform":
            mean = params.get("mean")
            half_range = params.get("half_range")
            if mean is None or half_range is None:
                raise ValueError("Uniform distribution requires 'mean' and 'half_range'.")
            low = mean - half_range
            high = mean + half_range
            description = f"Uniform({low:.2f}, {high:.2f})"

            def sampler(rng: random.Random) -> float:
                return rng.uniform(low, high)

            return Distribution(sampler, description)

        if dist_type == "triangular":
            low = params.get("min")
            mode = params.get("mode")
            high = params.get("max")
            if low is None or mode is None or high is None:
                raise ValueError("Triangular distribution requires 'min', 'mode', and 'max'.")
            description = f"Triangular({low:.2f}, {mode:.2f}, {high:.2f})"

            def sampler(rng: random.Random) -> float:
                return rng.triangular(low, high, mode)

            return Distribution(sampler, description)

        if dist_type == "normal":
            mean = params.get("mean")
            stdev = params.get("stdev")
            if mean is None or stdev is None:
                raise ValueError("Normal distribution requires 'mean' and 'stdev'.")
            description = f"Normal(mean={mean:.2f}, stdev={stdev:.2f})"

            def sampler(rng: random.Random) -> float:
                return rng.gauss(mean, stdev)

            return Distribution(sampler, description)

        if dist_type == "lognormal":
            mean = params.get("mean")
            stdev = params.get("stdev")
            if mean is None or stdev is None:
                raise ValueError("Lognormal distribution requires 'mean' and 'stdev'.")
            if mean <= 0:
                raise ValueError("Lognormal mean must be positive.")
            # Convert mean/stdev of raw distribution to mu/sigma of underlying normal
            variance = stdev ** 2
            phi = math.sqrt(variance + mean ** 2)
            sigma = math.sqrt(math.log((phi ** 2) / (mean ** 2)))
            mu = math.log(mean) - 0.5 * sigma ** 2
            description = f"LogNormal(mean={mean:.2f}, stdev={stdev:.2f})"

            def sampler(rng: random.Random) -> float:
                return rng.lognormvariate(mu, sigma)

            return Distribution(sampler, description)

        if dist_type == "exponential":
            mean = params.get("mean")
            if mean is None:
                raise ValueError("Exponential distribution requires 'mean'.")
            if mean <= 0:
                raise ValueError("Exponential mean must be positive.")
            lambd = 1.0 / mean
            description = f"Exponential(mean={mean:.2f})"

            def sampler(rng: random.Random) -> float:
                return rng.expovariate(lambd)

            return Distribution(sampler, description)

        raise ValueError(f"Unsupported distribution type '{config.type}'.")

    @staticmethod
    def from_dict(definition: Dict[str, float]) -> Distribution:
        dist_type = definition.get("type")
        if not dist_type:
            raise ValueError("Distribution definition requires a 'type' field.")
        params = {k: v for k, v in definition.items() if k != "type"}
        return DistributionFactory.from_config(DistributionConfig(type=dist_type, parameters=params))


def build_distribution_from_user_input(
    dist_type: str,
    mean: Optional[float] = None,
    half_range: Optional[float] = None,
    minimum: Optional[float] = None,
    mode: Optional[float] = None,
    maximum: Optional[float] = None,
    stdev: Optional[float] = None,
) -> Distribution:
    """Helper for legacy callers to create a distribution directly."""

    dist_type = dist_type.lower().strip()
    if dist_type == "uniform":
        if mean is None or half_range is None:
            raise ValueError("Uniform distribution requires mean and half_range.")
        return DistributionFactory.from_dict({"type": "uniform", "mean": mean, "half_range": half_range})
    if dist_type == "triangular":
        if minimum is None or mode is None or maximum is None:
            raise ValueError("Triangular distribution requires min, mode, and max.")
        return DistributionFactory.from_dict({"type": "triangular", "min": minimum, "mode": mode, "max": maximum})
    if dist_type == "normal":
        if mean is None or stdev is None:
            raise ValueError("Normal distribution requires mean and stdev.")
        return DistributionFactory.from_dict({"type": "normal", "mean": mean, "stdev": stdev})
    if dist_type == "lognormal":
        if mean is None or stdev is None:
            raise ValueError("Lognormal distribution requires mean and stdev.")
        return DistributionFactory.from_dict({"type": "lognormal", "mean": mean, "stdev": stdev})
    if dist_type == "exponential":
        if mean is None:
            raise ValueError("Exponential distribution requires mean.")
        return DistributionFactory.from_dict({"type": "exponential", "mean": mean})
    raise ValueError(f"Unsupported distribution type '{dist_type}'.")
