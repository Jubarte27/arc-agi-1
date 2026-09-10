"""Static Analysis Suite for ARC-AGI Benchmark Data.

Provides result-independent, model-agnostic, and task-agnostic structural analyses:
- Task Topology: Universally Solved vs Universally Unsolved Tasks & Inductive Traps
- Guttman Subsumption Hierarchy & 1D Scalogram Psychometrics
- Code Complexity & Occam's Razor Bloat Analysis
- Iteration Convergence Dynamics & Diminishing Returns
- Strategy Discrepancy, Complementarity & Oracle Ensemble Headroom
- Coordinate Leakage & Hardcoding Static Scanner
- Visualization engine for publication-grade figures
"""

from .task_topology import analyze_task_topology
from .guttman_hierarchy import analyze_guttman_hierarchy
from .code_complexity import analyze_code_complexity, extract_code_metrics
from .iteration_dynamics import analyze_iteration_dynamics
from .strategy_discrepancy import analyze_strategy_discrepancy
from .coordinate_leakage import analyze_coordinate_leakage
from .plotting import (
    plot_task_solve_distribution,
    plot_guttman_hierarchy,
    plot_code_complexity_by_outcome,
    plot_iteration_diminishing_returns,
    plot_strategy_discrepancy,
)

__all__ = [
    "analyze_task_topology",
    "analyze_guttman_hierarchy",
    "analyze_code_complexity",
    "extract_code_metrics",
    "analyze_iteration_dynamics",
    "analyze_strategy_discrepancy",
    "analyze_coordinate_leakage",
    "plot_task_solve_distribution",
    "plot_guttman_hierarchy",
    "plot_code_complexity_by_outcome",
    "plot_iteration_diminishing_returns",
    "plot_strategy_discrepancy",
]

