"""Monte Carlo utilities for running repeated simulations."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional

from .entities import ProcessLine
from .simulation import SimulationConfig, SimulationRunResult, simulate_line


@dataclass
class MonteCarloResult:
    """Stores the collection of individual simulation runs for a line."""

    line_name: str
    runs: List[SimulationRunResult]

    def metric_series(self, accessor) -> List[float]:
        return [accessor(run) for run in self.runs]


@dataclass
class MonteCarloSummary:
    """Aggregated Monte Carlo statistics."""

    line_name: str
    throughput_mean: float
    throughput_std: float
    cycle_time_mean: float
    cycle_time_std: float
    station_utilization: Dict[str, float]
    station_blocked: Dict[str, float]
    station_starved: Dict[str, float]

    @classmethod
    def from_result(cls, result: MonteCarloResult) -> "MonteCarloSummary":
        throughput_values = result.metric_series(lambda run: run.throughput_rate)
        cycle_values = result.metric_series(lambda run: run.average_cycle_time)
        throughput_mean = mean(throughput_values) if throughput_values else 0.0
        throughput_std = pstdev(throughput_values) if len(throughput_values) > 1 else 0.0
        cycle_mean = mean(cycle_values) if cycle_values else 0.0
        cycle_std = pstdev(cycle_values) if len(cycle_values) > 1 else 0.0

        station_utilization: Dict[str, List[float]] = {}
        station_blocked: Dict[str, List[float]] = {}
        station_starved: Dict[str, List[float]] = {}
        for run in result.runs:
            for stats in run.station_stats:
                station_utilization.setdefault(stats.name, []).append(stats.utilization)
                station_blocked.setdefault(stats.name, []).append(stats.blocked_ratio)
                station_starved.setdefault(stats.name, []).append(stats.starved_ratio)

        util_avg = {name: mean(values) for name, values in station_utilization.items()}
        blocked_avg = {name: mean(values) for name, values in station_blocked.items()}
        starved_avg = {name: mean(values) for name, values in station_starved.items()}

        return cls(
            line_name=result.line_name,
            throughput_mean=throughput_mean,
            throughput_std=throughput_std,
            cycle_time_mean=cycle_mean,
            cycle_time_std=cycle_std,
            station_utilization=util_avg,
            station_blocked=blocked_avg,
            station_starved=starved_avg,
        )


def run_monte_carlo(
    line: ProcessLine,
    jobs_to_complete: int,
    warmup_jobs: int,
    simulations: int,
    base_seed: Optional[int] = None,
) -> MonteCarloResult:
    runs: List[SimulationRunResult] = []
    for i in range(simulations):
        seed = None if base_seed is None else base_seed + i
        config = SimulationConfig(jobs_to_complete=jobs_to_complete, warmup_jobs=warmup_jobs, random_seed=seed)
        runs.append(simulate_line(line, config))
    return MonteCarloResult(line_name=line.name, runs=runs)


def summarize_results(results: Iterable[MonteCarloResult]) -> List[MonteCarloSummary]:
    return [MonteCarloSummary.from_result(result) for result in results]
