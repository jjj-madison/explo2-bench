"""Eight observation-only policies, public baseline revision 1."""
from .random_walk import RandomWalkAgent
from .random_revisit import RandomRevisitAgent
from .frontier_revisit import FrontierRevisitAgent
from .goal_revisit import GoalRevisitAgent
from .online_planner import OnlinePlannerAgent
from .sweep_then_exploit import SweepThenExploitAgent
from .utility import BalancedUtilityAgent, DependencyGreedyAgent

BASELINE_REVISION = 1
BASELINE_NAMES = (
    "random_walk", "random_revisit", "frontier_revisit", "goal_revisit",
    "sweep_then_exploit", "online_planner", "dependency_greedy", "balanced_utility",
)
STOCHASTIC_BASELINES = frozenset({"random_walk", "random_revisit"})


def available_agents() -> list[str]:
    return list(BASELINE_NAMES)


def create_agent(name: str, seed: int = 0, max_search_states: int = 10_000):
    """Build a policy. Search caps count generated states, including the root.

The cap applies independently to each search; aggregate work is reported in
``diagnostics``. No policy receives the hidden environment action budget.
"""
    if name not in BASELINE_NAMES:
        raise ValueError(f"Unknown baseline {name!r}; expected one of {BASELINE_NAMES}")
    if isinstance(max_search_states, bool) or not isinstance(max_search_states, int) or max_search_states < 1:
        raise ValueError("max_search_states must be a positive integer")
    factories = {
        "random_walk": lambda: RandomWalkAgent(seed),
        "random_revisit": lambda: RandomRevisitAgent(seed),
        "frontier_revisit": FrontierRevisitAgent,
        "goal_revisit": GoalRevisitAgent,
        "online_planner": lambda: OnlinePlannerAgent(max_search_states),
        "sweep_then_exploit": lambda: SweepThenExploitAgent(max_search_states),
        "dependency_greedy": DependencyGreedyAgent,
        "balanced_utility": BalancedUtilityAgent,
    }
    agent = factories[name]()
    if not hasattr(agent, "diagnostics"):
        agent.diagnostics = {}
    return agent


__all__ = ["BASELINE_REVISION", "BASELINE_NAMES", "STOCHASTIC_BASELINES", "available_agents", "create_agent"]
