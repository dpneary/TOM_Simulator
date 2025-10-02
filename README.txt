Process Flow Monte Carlo Simulator
==================================

This repository now includes an interactive application for analysing serial
process lines with Monte Carlo simulation. A polished graphical interface helps
non-technical stakeholders explore scenarios visually and compare performance at
a glance.

Quick start
-----------

1. Ensure you are using Python 3.9+.
2. Install the GUI dependency:

   ```bash
   pip install -r requirements.txt
   ```

3. From the repository root, launch the simulator:

   ```bash
   python run_simulator.py
   ```

4. Use the setup tab to choose the provided Line A/Line B examples or create your
   own process configuration. Adjust buffer capacities and Monte Carlo settings,
   then run the simulation to visualise throughput and workstation behaviour.

Capabilities
------------

* Configure multiple production lines side-by-side with an intuitive workstation
  table and buffer controls.
* Choose from several processing-time distributions (uniform, triangular, normal,
  lognormal, exponential) per workstation via guided dialogs.
* Control buffer capacities (0, finite, or effectively infinite) between
  workstations.
* Tune the Monte Carlo run length (warm-up jobs, observed jobs, number of
  replications, and random seed) directly from the setup screen.
* Review throughput and cycle-time summaries in an interactive results tab with
  comparison charts and per-station utilisation, blocked, and starved metrics.
* Run an automated buffer impact study that evaluates each potential buffer
  location and recommends the placement that yields the largest throughput gain.

Answering the prompt's question
-------------------------------

To replicate the example in the prompt:

1. Launch the simulator and choose the sample configurations for Line A and Line B.
2. Use the default simulation settings or adjust them as desired.
3. After the Monte Carlo results are displayed, use the buffer study controls on the
   results tab to evaluate adding a buffer and review the recommended location.

File overview
-------------

* `run_simulator.py` – entry point script.
* `simulator/` – Python package with the simulation engine and user interfaces.
  * `distributions.py` – user-friendly distribution helpers.
  * `entities.py` – task and process-line models.
  * `simulation.py` – discrete event simulation core.
  * `monte_carlo.py` – Monte Carlo orchestration helpers.
  * `gui.py` – Tkinter-based graphical interface with visual reporting.
  * `app.py` – original command-line interface (kept for reference).
