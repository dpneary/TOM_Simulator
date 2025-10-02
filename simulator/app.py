"""Interactive command-line application for the process flow simulator."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from .distributions import DistributionFactory
from .entities import ProcessLine, Task, infinite_buffer
from .monte_carlo import MonteCarloResult, MonteCarloSummary, run_monte_carlo, summarize_results


@dataclass
class LineInput:
    name: str
    tasks: List[Task]
    buffers: List[Optional[int]]


SAMPLE_LINES: Dict[str, LineInput] = {}


def _build_sample_lines() -> None:
    global SAMPLE_LINES
    if SAMPLE_LINES:
        return
    line_a_tasks = [
        Task(name=f"Task {i+1}", distribution=DistributionFactory.from_dict({"type": "uniform", "mean": 10, "half_range": 2}))
        for i in range(6)
    ]
    line_b_distributions = [
        {"type": "uniform", "mean": 6, "half_range": 2},
        {"type": "uniform", "mean": 10, "half_range": 2},
        {"type": "uniform", "mean": 4, "half_range": 3},
        {"type": "uniform", "mean": 7, "half_range": 1},
        {"type": "uniform", "mean": 5, "half_range": 3},
        {"type": "uniform", "mean": 5, "half_range": 2},
    ]
    line_b_tasks = [Task(name=f"Task {idx+1}", distribution=DistributionFactory.from_dict(dist)) for idx, dist in enumerate(line_b_distributions)]
    line_a = LineInput(name="Line A", tasks=line_a_tasks, buffers=[0] * (len(line_a_tasks) - 1))
    line_b = LineInput(name="Line B", tasks=line_b_tasks, buffers=[0] * (len(line_b_tasks) - 1))
    SAMPLE_LINES = {line_a.name: line_a, line_b.name: line_b}


def _input_with_default(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    response = input(f"{prompt}{suffix}: ").strip()
    if not response and default is not None:
        return default
    return response


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        response = _input_with_default(prompt, f"{default}")
        try:
            return float(response)
        except ValueError:
            print("Please enter a numeric value.")


def _prompt_int(prompt: str, default: int, minimum: int = 0) -> int:
    while True:
        response = _input_with_default(prompt, f"{default}")
        try:
            value = int(float(response))
            if value < minimum:
                raise ValueError
            return value
        except ValueError:
            print(f"Please enter an integer greater than or equal to {minimum}.")


def _prompt_choice(prompt: str, options: List[str], default: Optional[str] = None) -> str:
    option_map = {str(i + 1): opt for i, opt in enumerate(options)}
    while True:
        for idx, option in enumerate(options, start=1):
            marker = "" if default != option else " (default)"
            print(f"  {idx}. {option}{marker}")
        response = _input_with_default(prompt, None if default is None else str(options.index(default) + 1))
        chosen = option_map.get(response)
        if chosen:
            return chosen
        print("Please select one of the listed options by number.")


def _prompt_distribution(default_type: str = "uniform") -> Task:
    dist_types = ["Uniform", "Triangular", "Normal", "LogNormal", "Exponential"]
    chosen_type = _prompt_choice("Select distribution type", dist_types, default=default_type.title())
    dist_key = chosen_type.lower()
    if dist_key == "uniform":
        mean = _prompt_float("  Mean processing time", 10.0)
        half_range = _prompt_float("  Range (half-width)", 2.0)
        distribution = DistributionFactory.from_dict({"type": "uniform", "mean": mean, "half_range": half_range})
    elif dist_key == "triangular":
        minimum = _prompt_float("  Minimum", 5.0)
        mode = _prompt_float("  Most likely", (minimum + 10.0) / 2)
        maximum = _prompt_float("  Maximum", maximum := max(minimum + 1.0, mode + 1.0))
        distribution = DistributionFactory.from_dict({"type": "triangular", "min": minimum, "mode": mode, "max": maximum})
    elif dist_key == "normal":
        mean = _prompt_float("  Mean processing time", 10.0)
        stdev = _prompt_float("  Standard deviation", max(0.1, mean * 0.1))
        distribution = DistributionFactory.from_dict({"type": "normal", "mean": mean, "stdev": stdev})
    elif dist_key == "lognormal":
        mean = _prompt_float("  Mean processing time", 10.0)
        stdev = _prompt_float("  Standard deviation", max(0.1, mean * 0.2))
        distribution = DistributionFactory.from_dict({"type": "lognormal", "mean": mean, "stdev": stdev})
    else:
        mean = _prompt_float("  Mean processing time", 10.0)
        distribution = DistributionFactory.from_dict({"type": "exponential", "mean": mean})
    name = _input_with_default("  Task name", "Task")
    return Task(name=name or "Task", distribution=distribution)


def build_line_from_user(index: int) -> LineInput:
    print(f"\nConfiguring line #{index}...")
    name = _input_with_default("Line name", f"Line {index}")
    task_count = _prompt_int("How many workstations?", 3, minimum=1)
    tasks: List[Task] = []
    for task_index in range(task_count):
        print(f"Define workstation {task_index + 1}:")
        task = _prompt_distribution()
        if task.name == "Task":
            task.name = f"Task {task_index + 1}"
        tasks.append(task)
    buffers: List[Optional[int]] = []
    for buffer_index in range(task_count - 1):
        default_capacity = "0"
        response = _input_with_default(
            f"Buffer between {tasks[buffer_index].name} and {tasks[buffer_index + 1].name} (0, positive integer, or 'inf')",
            default_capacity,
        )
        if response.lower() in {"inf", "infinite", "infinity"}:
            buffers.append(infinite_buffer())
        else:
            try:
                capacity = int(float(response))
                if capacity < 0:
                    raise ValueError
                buffers.append(capacity)
            except ValueError:
                print("Invalid value, using 0 (no buffer).")
                buffers.append(0)
    return LineInput(name=name, tasks=tasks, buffers=buffers)


def select_line_from_samples() -> LineInput:
    _build_sample_lines()
    options = list(SAMPLE_LINES.keys())
    chosen = _prompt_choice("Choose a sample line", options, default=options[0])
    return SAMPLE_LINES[chosen]


def configure_lines() -> List[LineInput]:
    print("Welcome to the Process Flow Monte Carlo Simulator!")
    print("You can load one of the ready-to-run examples or create your own process line definition.\n")
    number_of_lines = _prompt_int("How many lines would you like to simulate?", 2, minimum=1)
    lines: List[LineInput] = []
    for idx in range(1, number_of_lines + 1):
        print(f"\nLine #{idx}")
        choice = _prompt_choice("Use a sample or build custom?", ["Sample line", "Create manually"], default="Sample line")
        if choice == "Sample line":
            selected = select_line_from_samples()
            lines.append(LineInput(name=selected.name, tasks=list(selected.tasks), buffers=list(selected.buffers)))
        else:
            lines.append(build_line_from_user(idx))
    return lines


def _format_value(value: float, unit: str = "") -> str:
    if math.isnan(value) or math.isinf(value):
        return "-"
    return f"{value:8.3f}{unit}"


def _print_line_summary(summary: MonteCarloSummary) -> None:
    print(f"\n=== {summary.line_name} ===")
    print(f"Throughput (jobs/min): {_format_value(summary.throughput_mean)} ± {_format_value(summary.throughput_std)}")
    print(f"Avg cycle time (min): {_format_value(summary.cycle_time_mean)} ± {_format_value(summary.cycle_time_std)}")
    print("Station performance:")
    header = f"{'Station':<20}{'Util%':>10}{'Blocked%':>12}{'Starved%':>12}"
    print(header)
    print("-" * len(header))
    for name in summary.station_utilization.keys():
        util = summary.station_utilization.get(name, 0.0) * 100
        blocked = summary.station_blocked.get(name, 0.0) * 100
        starved = summary.station_starved.get(name, 0.0) * 100
        print(f"{name:<20}{util:10.1f}{blocked:12.1f}{starved:12.1f}")


def _compare_throughput(summaries: List[MonteCarloSummary]) -> None:
    if len(summaries) < 2:
        return
    print("\nThroughput comparison (higher is better):")
    reference = max(summaries, key=lambda s: s.throughput_mean).throughput_mean
    for summary in summaries:
        delta = summary.throughput_mean - reference
        indicator = "*" if abs(delta) < 1e-6 else ("+" if delta > 0 else "-")
        print(f"  {summary.line_name:<20} {_format_value(summary.throughput_mean)} ({indicator}{delta:0.3f})")


def _run_buffer_study(line: LineInput, jobs: int, warmup: int, sims: int) -> None:
    if len(line.tasks) < 2:
        print("Buffer study requires at least two workstations.")
        return
    base_line = ProcessLine(name=line.name, tasks=line.tasks, buffer_capacities=line.buffers)
    baseline_result = MonteCarloSummary.from_result(run_monte_carlo(base_line, jobs, warmup, sims))
    print(f"\nBuffer impact study for {line.name}")
    base_throughput = baseline_result.throughput_mean
    print(f"  Baseline throughput: {base_throughput:.3f} jobs/min")
    candidate_capacity = _prompt_int("  Capacity of the new buffer (use a large number for \"infinite\")", 1, minimum=1)
    best_delta = float("-inf")
    best_location = None
    for idx in range(len(line.tasks) - 1):
        modified_buffers = list(line.buffers)
        modified_buffers[idx] = candidate_capacity
        modified_line = ProcessLine(name=f"{line.name} + buffer {idx+1}", tasks=line.tasks, buffer_capacities=modified_buffers)
        result = MonteCarloSummary.from_result(run_monte_carlo(modified_line, jobs, warmup, sims))
        delta = result.throughput_mean - base_throughput
        print(f"    Between {line.tasks[idx].name} and {line.tasks[idx+1].name}: throughput {result.throughput_mean:.3f} (Δ {delta:+.3f})")
        if delta > best_delta:
            best_delta = delta
            best_location = idx
    if best_location is None or best_delta <= 1e-6:
        print("  No buffer location provided a meaningful improvement.")
    else:
        print(
            f"  Suggested location: between {line.tasks[best_location].name} and {line.tasks[best_location + 1].name} "
            f"(Δ throughput {best_delta:+.3f} jobs/min)"
        )


def main() -> None:
    lines = configure_lines()
    jobs = _prompt_int("How many jobs should be observed (after warm-up)?", 500, minimum=1)
    warmup = _prompt_int("Warm-up jobs to discard?", 100, minimum=0)
    sims = _prompt_int("Number of Monte Carlo simulations?", 25, minimum=1)
    base_seed_input = _input_with_default("Random seed (leave blank for random)", "")
    base_seed = int(base_seed_input) if base_seed_input else None

    monte_results: List[MonteCarloResult] = []
    for line in lines:
        process_line = ProcessLine(name=line.name, tasks=line.tasks, buffer_capacities=line.buffers)
        result = run_monte_carlo(process_line, jobs_to_complete=jobs, warmup_jobs=warmup, simulations=sims, base_seed=base_seed)
        monte_results.append(result)

    summaries = summarize_results(monte_results)
    for summary in summaries:
        _print_line_summary(summary)
    _compare_throughput(summaries)

    for line in lines:
        response = _input_with_default(f"\nRun a buffer impact study for {line.name}? (y/n)", "n").lower()
        if response.startswith("y"):
            _run_buffer_study(line, jobs, warmup, sims)

    print("\nSimulation complete. Thank you for using the Process Flow Monte Carlo Simulator!")


if __name__ == "__main__":
    main()
