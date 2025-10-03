"""Core data structures for the process flow simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .distributions import Distribution, DistributionFactory


@dataclass
class Task:
    """Represents a single workstation or task in the process line."""

    name: str
    distribution: Distribution

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        distribution = DistributionFactory.from_dict(data["distribution"])
        return cls(name=data.get("name", "Task"), distribution=distribution)


@dataclass
class ProcessLine:
    """Represents a serial process line with intermediate buffers."""

    name: str
    tasks: Sequence[Task]
    buffer_capacities: Sequence[Optional[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        task_count = len(self.tasks)
        expected_buffers = max(0, task_count - 1)
        if len(self.buffer_capacities) != expected_buffers:
            if not self.buffer_capacities:
                self.buffer_capacities = [0 for _ in range(expected_buffers)]
            else:
                raise ValueError(
                    f"Process line '{self.name}' expected {expected_buffers} buffer capacities "
                    f"but received {len(self.buffer_capacities)}."
                )

    def describe(self) -> str:
        parts = []
        for idx, task in enumerate(self.tasks):
            parts.append(task.name)
            if idx < len(self.buffer_capacities):
                capacity = self.buffer_capacities[idx]
                if capacity is None:
                    parts.append("[∞ buffer]")
                else:
                    parts.append(f"[B{idx+1}:{capacity}]")
        return " -> ".join(parts)

    def with_buffer_override(self, index: int, capacity: Optional[int]) -> "ProcessLine":
        if index < 0 or index >= len(self.buffer_capacities):
            raise IndexError("Buffer index out of range.")
        new_buffers = list(self.buffer_capacities)
        new_buffers[index] = capacity
        return ProcessLine(name=f"{self.name} (buffer {index+1}={capacity})", tasks=self.tasks, buffer_capacities=new_buffers)


def infinite_buffer() -> Optional[int]:
    """Helper constant to represent an infinite buffer."""

    return None
