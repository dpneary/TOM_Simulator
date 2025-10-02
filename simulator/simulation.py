"""Discrete event simulation engine for the process flow lines."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import random
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

from .entities import ProcessLine


@dataclass
class StationStats:
    """Holds time-based statistics for a single station."""

    name: str
    processing_time: float
    blocked_time: float
    starved_time: float
    measured_time: float

    @property
    def utilization(self) -> float:
        return self.processing_time / self.measured_time if self.measured_time else 0.0

    @property
    def blocked_ratio(self) -> float:
        return self.blocked_time / self.measured_time if self.measured_time else 0.0

    @property
    def starved_ratio(self) -> float:
        return self.starved_time / self.measured_time if self.measured_time else 0.0


@dataclass
class SimulationRunResult:
    """Output of a single simulation run."""

    throughput_rate: float
    average_cycle_time: float
    completed_jobs: int
    measurement_start: float
    measurement_end: float
    station_stats: List[StationStats]

    @property
    def elapsed_time(self) -> float:
        return self.measurement_end - self.measurement_start


@dataclass
class _StationState:
    """Internal mutable state for each station during the simulation."""

    name: str
    status: str = "idle"  # idle, processing, blocked
    current_job: Optional[int] = None
    blocked_job: Optional[int] = None
    last_state_change: float = 0.0
    time_in_state: Dict[str, float] = None

    def __post_init__(self) -> None:
        if self.time_in_state is None:
            self.time_in_state = {"processing": 0.0, "blocked": 0.0, "idle": 0.0}


@dataclass
class _BufferState:
    capacity: Optional[int]
    queue: Deque[Tuple[int, float]]

    def __post_init__(self) -> None:
        if self.queue is None:
            self.queue = deque()

    def has_space(self) -> bool:
        if self.capacity is None:
            return True
        return len(self.queue) < self.capacity


class _EventQueue:
    def __init__(self) -> None:
        self._queue: List[Tuple[float, int, int, int]] = []
        self._counter = 0

    def push(self, time: float, station_index: int, job_id: int) -> None:
        heapq.heappush(self._queue, (time, self._counter, station_index, job_id))
        self._counter += 1

    def pop(self) -> Tuple[float, int, int, int]:
        return heapq.heappop(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)


@dataclass
class SimulationConfig:
    jobs_to_complete: int
    warmup_jobs: int
    random_seed: Optional[int] = None


def simulate_line(line: ProcessLine, config: SimulationConfig) -> SimulationRunResult:
    rng = random.Random(config.random_seed)
    station_states: List[_StationState] = [_StationState(task.name) for task in line.tasks]
    buffer_states: List[_BufferState] = [
        _BufferState(capacity=cap, queue=deque()) for cap in line.buffer_capacities
    ]
    events = _EventQueue()

    total_jobs_needed = config.jobs_to_complete + config.warmup_jobs
    jobs_started = 0
    jobs_completed = 0
    measurement_started = False
    measurement_start_time = 0.0
    last_completion_time = 0.0
    job_entry_times: Dict[int, float] = {}
    recorded_cycle_times: List[float] = []

    def record_state_duration(index: int, current_time: float) -> None:
        station = station_states[index]
        if not measurement_started:
            station.last_state_change = current_time
            return
        elapsed = current_time - station.last_state_change
        if elapsed < 0:
            return
        station.time_in_state[station.status] += elapsed
        station.last_state_change = current_time

    def change_status(index: int, new_status: str, current_time: float) -> None:
        record_state_duration(index, current_time)
        station_states[index].status = new_status
        station_states[index].last_state_change = current_time

    def start_processing(index: int, job_id: int, current_time: float) -> None:
        station = station_states[index]
        change_status(index, "processing", current_time)
        station.current_job = job_id
        duration = line.tasks[index].distribution.sample(rng)
        completion_time = current_time + max(0.0, duration)
        events.push(completion_time, index, job_id)

    def attempt_start(index: int, current_time: float) -> None:
        station = station_states[index]
        if station.status != "idle":
            return
        if index == 0:
            nonlocal jobs_started
            if jobs_started >= total_jobs_needed:
                return
            job_id = jobs_started
            jobs_started += 1
            job_entry_times[job_id] = current_time
            start_processing(index, job_id, current_time)
            return
        buffer = buffer_states[index - 1]
        if buffer.capacity == 0:
            prev_station = station_states[index - 1]
            if prev_station.status == "blocked" and prev_station.blocked_job is not None:
                job_id = prev_station.blocked_job
                prev_station.blocked_job = None
                change_status(index - 1, "idle", current_time)
                start_processing(index, job_id, current_time)
                attempt_start(index - 1, current_time)
            return
        if buffer.queue:
            job_id, _ = buffer.queue.popleft()
            start_processing(index, job_id, current_time)
            prev_station = station_states[index - 1]
            if prev_station.status == "blocked" and prev_station.blocked_job is not None and buffer.has_space():
                pending_job = prev_station.blocked_job
                prev_station.blocked_job = None
                change_status(index - 1, "idle", current_time)
                buffer.queue.append((pending_job, current_time))
                attempt_start(index - 1, current_time)
            return
        change_status(index, "idle", current_time)

    def release_blocked_station(index: int, current_time: float) -> None:
        station = station_states[index]
        if station.status != "blocked" or station.blocked_job is None:
            return
        next_index = index + 1
        buffer = buffer_states[index]
        if buffer.capacity == 0:
            next_station = station_states[next_index]
            if next_station.status == "idle":
                job_id = station.blocked_job
                station.blocked_job = None
                change_status(index, "idle", current_time)
                start_processing(next_index, job_id, current_time)
                attempt_start(index, current_time)
            return
        if buffer.has_space():
            job_id = station.blocked_job
            station.blocked_job = None
            change_status(index, "idle", current_time)
            buffer.queue.append((job_id, current_time))
            attempt_start(index, current_time)
            attempt_start(index + 1, current_time)

    def handle_completion(index: int, job_id: int, current_time: float) -> None:
        nonlocal jobs_completed, measurement_started, measurement_start_time, last_completion_time
        station = station_states[index]
        if station.current_job != job_id:
            return
        station.current_job = None
        change_status(index, "idle", current_time)
        next_index = index + 1
        if next_index >= len(station_states):
            jobs_completed += 1
            last_completion_time = current_time
            if jobs_completed == config.warmup_jobs:
                measurement_started = True
                measurement_start_time = current_time
                for idx in range(len(station_states)):
                    station_states[idx].time_in_state = {"processing": 0.0, "blocked": 0.0, "idle": 0.0}
                    station_states[idx].last_state_change = current_time
                    station_states[idx].status = station_states[idx].status
            if jobs_completed > config.warmup_jobs:
                cycle_time = current_time - job_entry_times[job_id]
                recorded_cycle_times.append(cycle_time)
            change_status(index, "idle", current_time)
            attempt_start(index, current_time)
            if index > 0 and line.buffer_capacities[index - 1] == 0:
                release_blocked_station(index - 1, current_time)
            return
        buffer = buffer_states[index]
        if buffer.capacity == 0:
            next_station = station_states[next_index]
            if next_station.status == "idle":
                start_processing(next_index, job_id, current_time)
                attempt_start(index, current_time)
            else:
                station.blocked_job = job_id
                change_status(index, "blocked", current_time)
            return
        if buffer.has_space():
            buffer.queue.append((job_id, current_time))
            attempt_start(next_index, current_time)
            attempt_start(index, current_time)
        else:
            station.blocked_job = job_id
            change_status(index, "blocked", current_time)

    attempt_start(0, 0.0)

    while events and jobs_completed < total_jobs_needed:
        time, _, station_index, job_id = events.pop()
        handle_completion(station_index, job_id, time)

    end_time = last_completion_time
    if not measurement_started:
        measurement_start_time = 0.0
        measurement_duration = end_time - measurement_start_time
    else:
        measurement_duration = end_time - measurement_start_time

    station_summaries: List[StationStats] = []
    for idx, station in enumerate(station_states):
        record_state_duration(idx, end_time)
        stats = StationStats(
            name=station.name,
            processing_time=station.time_in_state.get("processing", 0.0),
            blocked_time=station.time_in_state.get("blocked", 0.0),
            starved_time=station.time_in_state.get("idle", 0.0),
            measured_time=measurement_duration,
        )
        station_summaries.append(stats)

    throughput = 0.0
    avg_cycle_time = 0.0
    if measurement_duration > 0 and recorded_cycle_times:
        throughput = len(recorded_cycle_times) / measurement_duration
        avg_cycle_time = sum(recorded_cycle_times) / len(recorded_cycle_times)

    return SimulationRunResult(
        throughput_rate=throughput,
        average_cycle_time=avg_cycle_time,
        completed_jobs=len(recorded_cycle_times),
        measurement_start=measurement_start_time,
        measurement_end=end_time,
        station_stats=station_summaries,
    )
