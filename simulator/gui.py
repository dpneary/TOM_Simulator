"""Graphical user interface for the process flow simulator."""
from __future__ import annotations

import copy
import queue
import threading
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import messagebox, simpledialog
from typing import Dict, List, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from tkinter import ttk

from .distributions import DistributionConfig, DistributionFactory
from .entities import ProcessLine, Task, infinite_buffer
from .monte_carlo import MonteCarloResult, MonteCarloSummary, run_monte_carlo, summarize_results


@dataclass
class TaskDefinition:
    """Editable task definition backing the GUI."""

    name: str
    distribution_type: str
    parameters: Dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        try:
            distribution = DistributionFactory.from_config(
                DistributionConfig(type=self.distribution_type, parameters=self.parameters)
            )
            return distribution.description
        except Exception:
            return f"{self.distribution_type.title()} (?)"

    def to_task(self) -> Task:
        distribution = DistributionFactory.from_config(
            DistributionConfig(type=self.distribution_type, parameters=self.parameters)
        )
        return Task(name=self.name, distribution=distribution)


@dataclass
class LineDefinition:
    """Editable process line definition for the GUI."""

    name: str
    tasks: List[TaskDefinition]
    buffers: List[Optional[int]]

    def ensure_buffer_count(self) -> None:
        expected = max(0, len(self.tasks) - 1)
        if len(self.buffers) != expected:
            if len(self.buffers) < expected:
                self.buffers.extend([0] * (expected - len(self.buffers)))
            else:
                self.buffers = self.buffers[:expected]

    def to_process_line(self) -> ProcessLine:
        self.ensure_buffer_count()
        return ProcessLine(
            name=self.name,
            tasks=[task.to_task() for task in self.tasks],
            buffer_capacities=[None if buf is None else int(buf) for buf in self.buffers],
        )


def _build_sample_lines() -> Dict[str, LineDefinition]:
    uniform = lambda mean, half_range: {
        "type": "uniform",
        "mean": mean,
        "half_range": half_range,
    }
    line_a = LineDefinition(
        name="Line A",
        tasks=[
            TaskDefinition(name=f"Task {idx+1}", distribution_type="uniform", parameters=uniform(10, 2))
            for idx in range(6)
        ],
        buffers=[0] * 5,
    )
    line_b = LineDefinition(
        name="Line B",
        tasks=[
            TaskDefinition(name="Task 1", distribution_type="uniform", parameters=uniform(6, 2)),
            TaskDefinition(name="Task 2", distribution_type="uniform", parameters=uniform(10, 2)),
            TaskDefinition(name="Task 3", distribution_type="uniform", parameters=uniform(4, 3)),
            TaskDefinition(name="Task 4", distribution_type="uniform", parameters=uniform(7, 1)),
            TaskDefinition(name="Task 5", distribution_type="uniform", parameters=uniform(5, 3)),
            TaskDefinition(name="Task 6", distribution_type="uniform", parameters=uniform(5, 2)),
        ],
        buffers=[0] * 5,
    )
    return {line_a.name: line_a, line_b.name: line_b}


class TaskDialog(tk.Toplevel):
    """Dialog for creating or editing a task definition."""

    def __init__(self, parent: tk.Widget, task: Optional[TaskDefinition] = None) -> None:
        super().__init__(parent)
        self.title("Task settings")
        self.resizable(False, False)
        self.result: Optional[TaskDefinition] = None
        self.transient(parent)
        self.grab_set()

        self.name_var = tk.StringVar(value=task.name if task else "Task")
        self.type_var = tk.StringVar(value=(task.distribution_type if task else "uniform").lower())

        self.parameters: Dict[str, tk.StringVar] = {}

        content = ttk.Frame(self, padding=15)
        content.grid(row=0, column=0, sticky="nsew")

        ttk.Label(content, text="Task name:").grid(row=0, column=0, sticky="w")
        name_entry = ttk.Entry(content, textvariable=self.name_var, width=30)
        name_entry.grid(row=0, column=1, sticky="ew")
        name_entry.focus_set()

        ttk.Label(content, text="Distribution:").grid(row=1, column=0, sticky="w")
        combo = ttk.Combobox(
            content,
            textvariable=self.type_var,
            values=["uniform", "triangular", "normal", "lognormal", "exponential"],
            state="readonly",
        )
        combo.grid(row=1, column=1, sticky="ew")
        combo.bind("<<ComboboxSelected>>", lambda _event: self._render_parameter_fields())

        self.param_frame = ttk.LabelFrame(content, text="Parameters", padding=(10, 8))
        self.param_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        button_frame = ttk.Frame(content)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(15, 0))
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Save", command=self._on_save).grid(row=0, column=1, padx=5)

        self.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)

        self._render_parameter_fields(task)

    def _render_parameter_fields(self, task: Optional[TaskDefinition] = None) -> None:
        for child in self.param_frame.winfo_children():
            child.destroy()
        self.parameters.clear()

        dist_type = self.type_var.get().lower()
        defaults: Dict[str, float] = {}
        if task and task.distribution_type.lower() == dist_type:
            defaults = task.parameters
        field_specs = []
        if dist_type == "uniform":
            field_specs = [("Mean", "mean", defaults.get("mean", 10.0)), ("Half range", "half_range", defaults.get("half_range", 2.0))]
        elif dist_type == "triangular":
            field_specs = [
                ("Minimum", "min", defaults.get("min", 4.0)),
                ("Most likely", "mode", defaults.get("mode", 6.0)),
                ("Maximum", "max", defaults.get("max", 8.0)),
            ]
        elif dist_type == "normal":
            field_specs = [
                ("Mean", "mean", defaults.get("mean", 10.0)),
                ("Std dev", "stdev", defaults.get("stdev", 2.0)),
            ]
        elif dist_type == "lognormal":
            field_specs = [
                ("Mean", "mean", defaults.get("mean", 10.0)),
                ("Std dev", "stdev", defaults.get("stdev", 3.0)),
            ]
        else:
            field_specs = [("Mean", "mean", defaults.get("mean", 10.0))]

        for row, (label, key, value) in enumerate(field_specs):
            ttk.Label(self.param_frame, text=f"{label}:").grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=f"{value}")
            entry = ttk.Entry(self.param_frame, textvariable=var, width=18)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            self.parameters[key] = var
            if row == 0:
                entry.focus_set()

        self.param_frame.columnconfigure(1, weight=1)

    def _on_cancel(self) -> None:
        self.result = None
        self.destroy()

    def _on_save(self) -> None:
        name = self.name_var.get().strip() or "Task"
        dist_type = self.type_var.get().lower()
        params: Dict[str, float] = {}
        try:
            for key, var in self.parameters.items():
                params[key] = float(var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter numeric values for distribution parameters.", parent=self)
            return
        try:
            definition = TaskDefinition(name=name, distribution_type=dist_type, parameters=params)
            definition.describe()  # Validate configuration
        except Exception as exc:  # pragma: no cover - GUI validation
            messagebox.showerror("Invalid distribution", str(exc), parent=self)
            return
        self.result = definition
        self.destroy()


class SimulatorGUI:
    """Tkinter based application for configuring and running Monte Carlo studies."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Process Flow Monte Carlo Studio")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        self.lines: List[LineDefinition] = []
        self.selected_line_index: Optional[int] = None
        self.monte_carlo_settings = {
            "jobs": tk.StringVar(value="500"),
            "warmup": tk.StringVar(value="100"),
            "sims": tk.StringVar(value="25"),
            "seed": tk.StringVar(value=""),
        }

        self.status_var = tk.StringVar(value="Welcome! Configure your lines to begin.")

        self._simulation_thread: Optional[threading.Thread] = None
        self._simulation_queue: queue.Queue = queue.Queue()
        self._buffer_thread: Optional[threading.Thread] = None
        self._buffer_queue: queue.Queue = queue.Queue()

        self.monte_carlo_results: List[MonteCarloResult] = []
        self.monte_carlo_summaries: List[MonteCarloSummary] = []

        self._build_widgets()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.setup_frame = ttk.Frame(notebook, padding=15)
        self.results_frame = ttk.Frame(notebook, padding=15)
        notebook.add(self.setup_frame, text="Setup")
        notebook.add(self.results_frame, text="Results")

        self._build_setup_tab(self.setup_frame)
        self._build_results_tab(self.results_frame)

        status_bar = ttk.Frame(self.root)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(status_bar, textvariable=self.status_var, anchor="w", padding=(10, 5)).pack(fill=tk.X)

    def _build_setup_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        # Left column: list of lines and actions
        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        ttk.Label(left, text="Process lines", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        self.lines_list = tk.Listbox(left, height=20)
        self.lines_list.pack(fill=tk.BOTH, expand=True, pady=(6, 6))
        self.lines_list.bind("<<ListboxSelect>>", lambda _event: self._on_line_selected())

        button_frame = ttk.Frame(left)
        button_frame.pack(fill=tk.X)
        ttk.Button(button_frame, text="Add custom line", command=self._add_custom_line).grid(row=0, column=0, sticky="ew", pady=2)
        ttk.Button(button_frame, text="Add Line A example", command=lambda: self._add_sample_line("Line A")).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(button_frame, text="Add Line B example", command=lambda: self._add_sample_line("Line B")).grid(row=2, column=0, sticky="ew", pady=2)
        ttk.Button(button_frame, text="Remove selected", command=self._remove_selected_line).grid(row=3, column=0, sticky="ew", pady=2)
        button_frame.columnconfigure(0, weight=1)

        # Right column: line details
        details = ttk.Frame(parent)
        details.grid(row=0, column=1, sticky="nsew")
        details.columnconfigure(0, weight=1)
        details.rowconfigure(1, weight=1)

        header = ttk.Frame(details)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="Line name:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.line_name_var = tk.StringVar()
        self.line_name_entry = ttk.Entry(header, textvariable=self.line_name_var)
        self.line_name_entry.grid(row=0, column=1, sticky="ew")
        self.line_name_var.trace_add("write", lambda *_: self._update_line_name())

        task_frame = ttk.LabelFrame(details, text="Workstations", padding=10)
        task_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 12))
        task_frame.columnconfigure(0, weight=1)
        task_frame.rowconfigure(0, weight=1)

        columns = ("task", "distribution", "buffer")
        self.task_tree = ttk.Treeview(task_frame, columns=columns, show="headings", selectmode="browse")
        self.task_tree.heading("task", text="Workstation")
        self.task_tree.heading("distribution", text="Distribution")
        self.task_tree.heading("buffer", text="Buffer after")
        self.task_tree.column("task", width=180, anchor="w")
        self.task_tree.column("distribution", width=280, anchor="w")
        self.task_tree.column("buffer", width=120, anchor="center")
        self.task_tree.grid(row=0, column=0, sticky="nsew")

        tree_scroll = ttk.Scrollbar(task_frame, orient=tk.VERTICAL, command=self.task_tree.yview)
        self.task_tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.grid(row=0, column=1, sticky="ns")

        actions = ttk.Frame(task_frame)
        actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        actions.columnconfigure((0, 1, 2, 3, 4, 5), weight=1)
        ttk.Button(actions, text="Add task", command=self._add_task).grid(row=0, column=0, padx=2)
        ttk.Button(actions, text="Edit task", command=self._edit_task).grid(row=0, column=1, padx=2)
        ttk.Button(actions, text="Remove task", command=self._remove_task).grid(row=0, column=2, padx=2)
        ttk.Button(actions, text="Move up", command=lambda: self._move_task(-1)).grid(row=0, column=3, padx=2)
        ttk.Button(actions, text="Move down", command=lambda: self._move_task(1)).grid(row=0, column=4, padx=2)
        ttk.Button(actions, text="Edit buffer", command=self._edit_buffer).grid(row=0, column=5, padx=2)

        sim_frame = ttk.LabelFrame(details, text="Monte Carlo settings", padding=10)
        sim_frame.grid(row=2, column=0, sticky="ew")
        for col in range(8):
            sim_frame.columnconfigure(col, weight=1)

        ttk.Label(sim_frame, text="Jobs to complete:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Entry(sim_frame, textvariable=self.monte_carlo_settings["jobs"], width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(sim_frame, text="Warm-up jobs:").grid(row=0, column=2, sticky="w", padx=(12, 4))
        ttk.Entry(sim_frame, textvariable=self.monte_carlo_settings["warmup"], width=10).grid(row=0, column=3, sticky="w")
        ttk.Label(sim_frame, text="Simulations:").grid(row=0, column=4, sticky="w", padx=(12, 4))
        ttk.Entry(sim_frame, textvariable=self.monte_carlo_settings["sims"], width=10).grid(row=0, column=5, sticky="w")
        ttk.Label(sim_frame, text="Random seed:").grid(row=0, column=6, sticky="w", padx=(12, 4))
        ttk.Entry(sim_frame, textvariable=self.monte_carlo_settings["seed"], width=10).grid(row=0, column=7, sticky="w")

        ttk.Button(details, text="Run simulation", command=self._run_simulation).grid(row=3, column=0, sticky="e", pady=(10, 0))

    def _build_results_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        intro = ttk.Label(
            parent,
            text="Run a simulation to view throughput and workstation performance summaries.",
            font=("Segoe UI", 11),
        )
        intro.grid(row=0, column=0, sticky="w")

        summary_frame = ttk.Frame(parent)
        summary_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.columnconfigure(1, weight=1)
        summary_frame.rowconfigure(1, weight=1)

        tree_container = ttk.LabelFrame(summary_frame, text="Line summaries", padding=10)
        tree_container.grid(row=0, column=0, sticky="nsew")
        tree_container.columnconfigure(0, weight=1)
        tree_container.rowconfigure(0, weight=1)

        columns = ("throughput", "throughput_std", "cycle", "cycle_std")
        self.summary_tree = ttk.Treeview(tree_container, columns=columns, show="headings", selectmode="browse")
        self.summary_tree.heading("throughput", text="Throughput (jobs/min)")
        self.summary_tree.heading("throughput_std", text="σ throughput")
        self.summary_tree.heading("cycle", text="Avg cycle time (min)")
        self.summary_tree.heading("cycle_std", text="σ cycle")
        self.summary_tree.column("throughput", width=160, anchor="center")
        self.summary_tree.column("throughput_std", width=120, anchor="center")
        self.summary_tree.column("cycle", width=150, anchor="center")
        self.summary_tree.column("cycle_std", width=100, anchor="center")
        self.summary_tree.grid(row=0, column=0, sticky="nsew")

        summary_scroll = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.summary_tree.yview)
        self.summary_tree.configure(yscrollcommand=summary_scroll.set)
        summary_scroll.grid(row=0, column=1, sticky="ns")

        chart_frame = ttk.LabelFrame(summary_frame, text="Throughput comparison", padding=10)
        chart_frame.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(5.5, 3.8), dpi=100)
        self.throughput_ax = self.figure.add_subplot(111)
        self.throughput_ax.set_xlabel("Process line")
        self.throughput_ax.set_ylabel("Throughput (jobs/min)")
        self.throughput_canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.throughput_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        detail_frame = ttk.LabelFrame(summary_frame, text="Line details", padding=10)
        detail_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(2, weight=1)

        selector_frame = ttk.Frame(detail_frame)
        selector_frame.grid(row=0, column=0, sticky="ew")
        ttk.Label(selector_frame, text="Select line:").grid(row=0, column=0, sticky="w")
        self.detail_line_var = tk.StringVar()
        self.detail_line_combo = ttk.Combobox(selector_frame, textvariable=self.detail_line_var, state="readonly")
        self.detail_line_combo.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.detail_line_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_station_details())

        self.detail_summary_var = tk.StringVar()
        ttk.Label(detail_frame, textvariable=self.detail_summary_var, font=("Segoe UI", 10, "bold")).grid(
            row=1, column=0, sticky="w", pady=(6, 6)
        )

        columns = ("station", "util", "blocked", "starved")
        self.station_tree = ttk.Treeview(detail_frame, columns=columns, show="headings")
        self.station_tree.heading("station", text="Workstation")
        self.station_tree.heading("util", text="Utilization %")
        self.station_tree.heading("blocked", text="Blocked %")
        self.station_tree.heading("starved", text="Starved %")
        self.station_tree.column("station", width=200, anchor="w")
        self.station_tree.column("util", width=120, anchor="center")
        self.station_tree.column("blocked", width=120, anchor="center")
        self.station_tree.column("starved", width=120, anchor="center")
        self.station_tree.grid(row=2, column=0, sticky="nsew")

        buffer_frame = ttk.Frame(detail_frame)
        buffer_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(buffer_frame, text="Buffer capacity to evaluate:").grid(row=0, column=0, sticky="w")
        self.buffer_capacity_var = tk.StringVar(value="1")
        ttk.Entry(buffer_frame, textvariable=self.buffer_capacity_var, width=8).grid(row=0, column=1, sticky="w", padx=(4, 12))
        ttk.Button(buffer_frame, text="Suggest best location", command=self._suggest_buffer_location).grid(row=0, column=2, sticky="w")
        self.buffer_suggestion_var = tk.StringVar()
        ttk.Label(buffer_frame, textvariable=self.buffer_suggestion_var, wraplength=800, justify="left").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )

    # ------------------------------------------------------------------
    # Line management helpers
    # ------------------------------------------------------------------
    def _add_custom_line(self) -> None:
        default_tasks = [
            TaskDefinition(name="Task 1", distribution_type="uniform", parameters={"mean": 10.0, "half_range": 2.0}),
            TaskDefinition(name="Task 2", distribution_type="uniform", parameters={"mean": 10.0, "half_range": 2.0}),
            TaskDefinition(name="Task 3", distribution_type="uniform", parameters={"mean": 10.0, "half_range": 2.0}),
        ]
        line = LineDefinition(name=f"Line {len(self.lines) + 1}", tasks=default_tasks, buffers=[0, 0])
        self.lines.append(line)
        self._refresh_line_list(select=len(self.lines) - 1)
        self.status_var.set("Added a new custom line. Edit the workstations to match your process.")

    def _add_sample_line(self, sample_name: str) -> None:
        samples = _build_sample_lines()
        line = samples.get(sample_name)
        if not line:
            messagebox.showerror("Sample not found", f"Sample line '{sample_name}' is unavailable.")
            return
        cloned = copy.deepcopy(line)
        self.lines.append(cloned)
        self._refresh_line_list(select=len(self.lines) - 1)
        self.status_var.set(f"Loaded {sample_name}. Adjust as needed before running the simulation.")

    def _remove_selected_line(self) -> None:
        index = self.selected_line_index
        if index is None:
            return
        removed = self.lines.pop(index)
        self.selected_line_index = None
        self._refresh_line_list()
        self.status_var.set(f"Removed {removed.name}.")

    def _refresh_line_list(self, select: Optional[int] = None) -> None:
        self.lines_list.delete(0, tk.END)
        for line in self.lines:
            self.lines_list.insert(tk.END, line.name)
        if select is not None and 0 <= select < len(self.lines):
            self.lines_list.selection_clear(0, tk.END)
            self.lines_list.selection_set(select)
            self.lines_list.activate(select)
            self.selected_line_index = select
            self._on_line_selected()
        elif self.lines:
            self.lines_list.selection_set(0)
            self.lines_list.activate(0)
            self.selected_line_index = 0
            self._on_line_selected()
        else:
            self.selected_line_index = None
            self._clear_line_details()

    def _clear_line_details(self) -> None:
        self.line_name_var.set("")
        for row in self.task_tree.get_children():
            self.task_tree.delete(row)

    def _on_line_selected(self) -> None:
        selection = self.lines_list.curselection()
        if not selection:
            self.selected_line_index = None
            self._clear_line_details()
            return
        index = selection[0]
        self.selected_line_index = index
        line = self.lines[index]
        self.line_name_var.set(line.name)
        self._refresh_task_tree(line)

    def _refresh_task_tree(self, line: LineDefinition) -> None:
        for row in self.task_tree.get_children():
            self.task_tree.delete(row)
        line.ensure_buffer_count()
        for idx, task in enumerate(line.tasks):
            buffer_value = "" if idx >= len(line.buffers) else self._format_buffer(line.buffers[idx])
            self.task_tree.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(task.name, task.describe(), buffer_value),
            )

    def _update_line_name(self) -> None:
        if self.selected_line_index is None:
            return
        line = self.lines[self.selected_line_index]
        line.name = self.line_name_var.get().strip() or f"Line {self.selected_line_index + 1}"
        self.lines_list.delete(self.selected_line_index)
        self.lines_list.insert(self.selected_line_index, line.name)
        self.lines_list.selection_set(self.selected_line_index)
        self.lines_list.activate(self.selected_line_index)
        self.status_var.set(f"Updated line name to {line.name}.")

    def _add_task(self) -> None:
        if self.selected_line_index is None:
            return
        dialog = TaskDialog(self.root)
        self.root.wait_window(dialog)
        if dialog.result:
            line = self.lines[self.selected_line_index]
            line.tasks.append(dialog.result)
            line.ensure_buffer_count()
            self._refresh_task_tree(line)
            self.status_var.set(f"Added {dialog.result.name} to {line.name}.")

    def _edit_task(self) -> None:
        if self.selected_line_index is None:
            return
        selected = self.task_tree.selection()
        if not selected:
            messagebox.showinfo("Select a task", "Please choose a workstation to edit.")
            return
        index = int(selected[0])
        line = self.lines[self.selected_line_index]
        dialog = TaskDialog(self.root, line.tasks[index])
        self.root.wait_window(dialog)
        if dialog.result:
            line.tasks[index] = dialog.result
            self._refresh_task_tree(line)
            self.status_var.set(f"Updated {dialog.result.name}.")

    def _remove_task(self) -> None:
        if self.selected_line_index is None:
            return
        selected = self.task_tree.selection()
        if not selected:
            return
        index = int(selected[0])
        line = self.lines[self.selected_line_index]
        if len(line.tasks) <= 1:
            messagebox.showinfo("Cannot remove", "A line must have at least one workstation.")
            return
        removed = line.tasks.pop(index)
        line.ensure_buffer_count()
        self._refresh_task_tree(line)
        self.status_var.set(f"Removed {removed.name}.")

    def _move_task(self, direction: int) -> None:
        if self.selected_line_index is None:
            return
        selected = self.task_tree.selection()
        if not selected:
            return
        index = int(selected[0])
        new_index = index + direction
        line = self.lines[self.selected_line_index]
        if new_index < 0 or new_index >= len(line.tasks):
            return
        line.tasks[index], line.tasks[new_index] = line.tasks[new_index], line.tasks[index]
        line.ensure_buffer_count()
        self._refresh_task_tree(line)
        self.task_tree.selection_set(str(new_index))
        self.status_var.set(f"Moved {line.tasks[new_index].name}.")

    def _edit_buffer(self) -> None:
        if self.selected_line_index is None:
            return
        selected = self.task_tree.selection()
        if not selected:
            messagebox.showinfo("Select location", "Choose the workstation before the buffer to edit.")
            return
        index = int(selected[0])
        line = self.lines[self.selected_line_index]
        line.ensure_buffer_count()
        if index >= len(line.buffers):
            messagebox.showinfo("No buffer", "There is no buffer after the last workstation.")
            return
        current = line.buffers[index]
        default = "inf" if current is None else str(current)
        response = simpledialog.askstring(
            "Buffer capacity",
            "Enter 0 for none, a positive integer for limited storage, or 'inf' for unlimited:",
            initialvalue=default,
            parent=self.root,
        )
        if response is None:
            return
        response = response.strip().lower()
        if response in {"inf", "infinite", "infinity"}:
            line.buffers[index] = infinite_buffer()
        else:
            try:
                value = int(float(response))
                if value < 0:
                    raise ValueError
                line.buffers[index] = value
            except ValueError:
                messagebox.showerror("Invalid value", "Please enter 'inf' or a non-negative integer.")
                return
        self._refresh_task_tree(line)
        self.status_var.set(
            f"Updated buffer between {line.tasks[index].name} and {line.tasks[index + 1].name} to {self._format_buffer(line.buffers[index])}."
        )

    # ------------------------------------------------------------------
    # Simulation execution
    # ------------------------------------------------------------------
    def _parse_positive_int(self, value: str, field: str, minimum: int = 1) -> Optional[int]:
        try:
            parsed = int(float(value))
            if parsed < minimum:
                raise ValueError
            return parsed
        except ValueError:
            messagebox.showerror("Invalid input", f"{field} must be an integer ≥ {minimum}.")
            return None

    def _parse_non_negative_int(self, value: str, field: str) -> Optional[int]:
        try:
            parsed = int(float(value))
            if parsed < 0:
                raise ValueError
            return parsed
        except ValueError:
            messagebox.showerror("Invalid input", f"{field} must be a non-negative integer.")
            return None

    def _run_simulation(self) -> None:
        if self._simulation_thread and self._simulation_thread.is_alive():
            return
        if not self.lines:
            messagebox.showinfo("Add a line", "Please add at least one process line before running a simulation.")
            return
        jobs = self._parse_positive_int(self.monte_carlo_settings["jobs"].get(), "Jobs to complete")
        if jobs is None:
            return
        warmup = self._parse_non_negative_int(self.monte_carlo_settings["warmup"].get(), "Warm-up jobs")
        if warmup is None:
            return
        sims = self._parse_positive_int(self.monte_carlo_settings["sims"].get(), "Simulations")
        if sims is None:
            return
        seed_text = self.monte_carlo_settings["seed"].get().strip()
        seed = None
        if seed_text:
            try:
                seed = int(seed_text)
            except ValueError:
                messagebox.showerror("Invalid seed", "Random seed must be an integer or left blank.")
                return

        self.status_var.set("Running simulations...")
        self._simulation_queue = queue.Queue()
        self._simulation_thread = threading.Thread(
            target=self._execute_simulation,
            args=(jobs, warmup, sims, seed),
            daemon=True,
        )
        self._simulation_thread.start()
        self.root.after(100, self._check_simulation)

    def _execute_simulation(self, jobs: int, warmup: int, sims: int, seed: Optional[int]) -> None:
        results: List[MonteCarloResult] = []
        try:
            for line in self.lines:
                process_line = line.to_process_line()
                result = run_monte_carlo(
                    process_line,
                    jobs_to_complete=jobs,
                    warmup_jobs=warmup,
                    simulations=sims,
                    base_seed=seed,
                )
                results.append(result)
        except Exception as exc:  # pragma: no cover - safeguards runtime errors
            self._simulation_queue.put(exc)
            return
        self._simulation_queue.put(results)

    def _check_simulation(self) -> None:
        if self._simulation_thread and self._simulation_thread.is_alive():
            self.root.after(100, self._check_simulation)
            return
        try:
            payload = self._simulation_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._check_simulation)
            return
        if isinstance(payload, Exception):
            messagebox.showerror("Simulation error", str(payload))
            self.status_var.set("Simulation failed. Adjust your configuration and try again.")
            return
        self.monte_carlo_results = payload
        self.monte_carlo_summaries = summarize_results(self.monte_carlo_results)
        self._refresh_results_tab()
        self.status_var.set("Simulation complete. Review the results tab for insights.")

    # ------------------------------------------------------------------
    # Results presentation
    # ------------------------------------------------------------------
    def _refresh_results_tab(self) -> None:
        for row in self.summary_tree.get_children():
            self.summary_tree.delete(row)
        names = []
        throughputs = []
        errors = []
        for summary in self.monte_carlo_summaries:
            names.append(summary.line_name)
            throughputs.append(summary.throughput_mean)
            errors.append(summary.throughput_std)
            self.summary_tree.insert(
                "",
                tk.END,
                iid=summary.line_name,
                values=(
                    f"{summary.throughput_mean:.3f}",
                    f"{summary.throughput_std:.3f}",
                    f"{summary.cycle_time_mean:.3f}",
                    f"{summary.cycle_time_std:.3f}",
                ),
            )
        self._update_throughput_chart(names, throughputs, errors)
        self.detail_line_combo["values"] = names
        if names:
            self.detail_line_combo.current(0)
            self._refresh_station_details()
        else:
            self.detail_line_combo.set("")
            self.detail_summary_var.set("")
            for row in self.station_tree.get_children():
                self.station_tree.delete(row)

    def _update_throughput_chart(self, names: List[str], values: List[float], errors: List[float]) -> None:
        self.throughput_ax.clear()
        if names:
            indices = range(len(names))
            bars = self.throughput_ax.bar(indices, values, yerr=errors, capsize=6, color="#4C78A8")
            self.throughput_ax.set_xticks(indices)
            self.throughput_ax.set_xticklabels(names, rotation=15, ha="right")
            self.throughput_ax.set_ylabel("Throughput (jobs/min)")
            self.throughput_ax.yaxis.set_major_locator(MaxNLocator(5))
            for bar, value in zip(bars, values):
                height = bar.get_height()
                self.throughput_ax.annotate(
                    f"{value:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        else:
            self.throughput_ax.text(0.5, 0.5, "Run a simulation to view results", ha="center", va="center")
        self.throughput_ax.margins(x=0.05)
        self.throughput_canvas.draw_idle()

    def _refresh_station_details(self) -> None:
        line_name = self.detail_line_var.get()
        summary = next((s for s in self.monte_carlo_summaries if s.line_name == line_name), None)
        if not summary:
            return
        self.detail_summary_var.set(
            f"Throughput {summary.throughput_mean:.3f} jobs/min (σ {summary.throughput_std:.3f}) | "
            f"Cycle time {summary.cycle_time_mean:.3f} min"
        )
        for row in self.station_tree.get_children():
            self.station_tree.delete(row)
        for station, util in summary.station_utilization.items():
            blocked = summary.station_blocked.get(station, 0.0)
            starved = summary.station_starved.get(station, 0.0)
            self.station_tree.insert(
                "",
                tk.END,
                values=(
                    station,
                    f"{util * 100:.1f}",
                    f"{blocked * 100:.1f}",
                    f"{starved * 100:.1f}",
                ),
            )

    # ------------------------------------------------------------------
    # Buffer suggestion helper
    # ------------------------------------------------------------------
    def _suggest_buffer_location(self) -> None:
        if self._buffer_thread and self._buffer_thread.is_alive():
            return
        if self.selected_line_index is None:
            messagebox.showinfo("Select a line", "Select a line on the setup tab to evaluate buffer locations.")
            return
        line = self.lines[self.selected_line_index]
        if len(line.tasks) < 2:
            messagebox.showinfo("Not enough tasks", "Add at least two workstations to analyze buffer placement.")
            return
        capacity_text = self.buffer_capacity_var.get().strip()
        try:
            capacity = int(float(capacity_text))
            if capacity <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid capacity", "Enter a positive integer capacity for the new buffer.")
            return
        jobs = self._parse_positive_int(self.monte_carlo_settings["jobs"].get(), "Jobs to complete")
        warmup = self._parse_non_negative_int(self.monte_carlo_settings["warmup"].get(), "Warm-up jobs")
        sims = self._parse_positive_int(self.monte_carlo_settings["sims"].get(), "Simulations")
        if jobs is None or warmup is None or sims is None:
            return
        seed_text = self.monte_carlo_settings["seed"].get().strip()
        seed = int(seed_text) if seed_text else None

        self.buffer_suggestion_var.set("Evaluating buffer locations...")
        self._buffer_queue = queue.Queue()
        self._buffer_thread = threading.Thread(
            target=self._execute_buffer_study,
            args=(line, capacity, jobs, warmup, sims, seed),
            daemon=True,
        )
        self._buffer_thread.start()
        self.root.after(100, self._check_buffer_study)

    def _execute_buffer_study(
        self,
        line: LineDefinition,
        capacity: int,
        jobs: int,
        warmup: int,
        sims: int,
        seed: Optional[int],
    ) -> None:
        try:
            base_line = line.to_process_line()
            base_summary = MonteCarloSummary.from_result(
                run_monte_carlo(base_line, jobs_to_complete=jobs, warmup_jobs=warmup, simulations=sims, base_seed=seed)
            )
            best_delta = float("-inf")
            best_index = None
            messages = []
            for idx in range(len(line.tasks) - 1):
                modified = base_line.with_buffer_override(idx, capacity)
                summary = MonteCarloSummary.from_result(
                    run_monte_carlo(modified, jobs_to_complete=jobs, warmup_jobs=warmup, simulations=sims, base_seed=seed)
                )
                delta = summary.throughput_mean - base_summary.throughput_mean
                before = line.tasks[idx].name
                after = line.tasks[idx + 1].name
                messages.append(
                    f"Between {before} and {after}: throughput {summary.throughput_mean:.3f} (Δ {delta:+.3f})"
                )
                if delta > best_delta:
                    best_delta = delta
                    best_index = idx
            if best_index is None or best_delta <= 1e-6:
                result_text = "No buffer location provided a meaningful improvement."
            else:
                before = line.tasks[best_index].name
                after = line.tasks[best_index + 1].name
                result_text = (
                    f"Suggested location: between {before} and {after} (Δ throughput {best_delta:+.3f} jobs/min)."
                )
            final = "\n".join(messages + ["", result_text])
            self._buffer_queue.put(final)
        except Exception as exc:  # pragma: no cover - runtime guard
            self._buffer_queue.put(f"Buffer study failed: {exc}")

    def _check_buffer_study(self) -> None:
        if self._buffer_thread and self._buffer_thread.is_alive():
            self.root.after(100, self._check_buffer_study)
            return
        try:
            message = self._buffer_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._check_buffer_study)
            return
        self.buffer_suggestion_var.set(message)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _format_buffer(value: Optional[int]) -> str:
        if value is None:
            return "∞"
        return str(value)


def launch() -> None:
    root = tk.Tk()
    SimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover - manual execution
    launch()
