"""Analyze shortest solutions by their rule-manipulation strategy.

The game admits many action sequences that differ only in how the controlled
object walks through empty cells.  This module collapses those navigation-only
differences and compares solutions through semantic events instead:

* which rule words were pushed;
* which active rules were added or removed; and
* which controlled object reached a WIN object.

It is intentionally independent from the offline ``reference_solution`` when
searching.  The reference is replayed only after the complete shortest-solution
layer has been enumerated, so it can be compared with the discovered strategy
set rather than used to guide the search.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import baba
from baba.world_object import RuleColor, RuleIs, RuleObject, RuleProperty


ACTION_NAMES = ("up", "right", "down", "left")

DEFAULT_TEMPLATE_SEEDS = {
    "env/composition-01-target-lock": (0, 2, 3, 6),
    "env/composition-02-stop-to-win": (0, 2, 3, 6),
    "env/composition-03-reuse-is": (0, 3, 7, 18),
    "env/composition-04-reuse-win": (0, 3, 7, 18),
    "env/composition-05-cross-rule": (0, 3, 7, 18),
    "env/composition-06-color-scope": (0, 1, 3, 7),
    "env/composition-07-one-token-fork": (0, 2, 3, 6),
    "env/composition-08-control-handoff": (0, 1, 3, 7),
    "env/composition-09-ordered-assembly": (0, 2, 3, 6),
    "env/composition-10-control-relay": (0, 1, 3, 7),
}

PROPERTY_NAMES = {
    "is_agent": "YOU",
    "is_goal": "WIN",
    "is_stop": "STOP",
    "is_push": "PUSH",
    "is_move": "MOVE",
    "is_pull": "PULL",
    "is_defeat": "DEFEAT",
}

EventGroup = Tuple[str, ...]
SemanticTrace = Tuple[EventGroup, ...]


class RoleNames:
    """Map seed-specific object names to stable roles such as TARGET."""

    ROLE_FIELDS = (
        "target",
        "blocker",
        "goal",
        "wrong_target",
        "decoy",
        "distractor",
    )

    def __init__(self, generation_params: Mapping[str, object]):
        self.object_roles = {"baba": "BABA", "fwall": "WALL"}
        for field_name in self.ROLE_FIELDS:
            value = generation_params.get(field_name)
            if value is not None:
                self.object_roles.setdefault(f"f{value}", field_name.upper())

    def object(self, object_type: str) -> str:
        if object_type in self.object_roles:
            return self.object_roles[object_type]
        return object_type.removeprefix("f").upper()

    @staticmethod
    def property(property_name: str) -> str:
        return PROPERTY_NAMES.get(property_name, property_name.removeprefix("is_").upper())


@dataclass(frozen=True)
class SemanticSnapshot:
    rules: frozenset[str]
    rule_tokens: Mapping[str, Counter]


@dataclass
class SearchNode:
    depth: int
    env: object
    action_path_count: int = 0
    trace_examples: Dict[SemanticTrace, Tuple[str, ...]] = field(default_factory=dict)


@dataclass
class ShortestSolutionAnalysis:
    task: str
    seed: int
    variant: int
    reference_length: int
    shortest_length: int
    shortest_action_sequence_count: int
    semantic_trace_count: int
    semantic_traces: Tuple[SemanticTrace, ...]
    semantic_trace_examples: Tuple[Tuple[str, ...], ...]
    rule_strategy_trace_count: int
    rule_strategy_traces: Tuple[SemanticTrace, ...]
    reference_trace: SemanticTrace
    reference_trace_is_shortest: bool
    expanded_states: int

    @property
    def shortest_strategy_is_unique(self) -> bool:
        return self.semantic_trace_count == 1

    @property
    def shortest_rule_strategy_is_unique(self) -> bool:
        return self.rule_strategy_trace_count == 1

    def to_dict(self) -> dict:
        def trace_to_json(trace: SemanticTrace):
            return [list(event_group) for event_group in trace]

        return {
            "task": self.task,
            "seed": self.seed,
            "variant": self.variant,
            "reference_length": self.reference_length,
            "shortest_length": self.shortest_length,
            "shortest_action_sequence_count": self.shortest_action_sequence_count,
            "semantic_trace_count": self.semantic_trace_count,
            "shortest_strategy_is_unique": self.shortest_strategy_is_unique,
            "semantic_traces": [trace_to_json(trace) for trace in self.semantic_traces],
            "semantic_trace_examples": [list(actions) for actions in self.semantic_trace_examples],
            "rule_strategy_trace_count": self.rule_strategy_trace_count,
            "shortest_rule_strategy_is_unique": self.shortest_rule_strategy_is_unique,
            "rule_strategy_traces": [trace_to_json(trace) for trace in self.rule_strategy_traces],
            "reference_trace": trace_to_json(self.reference_trace),
            "reference_trace_is_shortest": self.reference_trace_is_shortest,
            "expanded_states": self.expanded_states,
        }


@dataclass
class CounterfactualSearchResult:
    kind: str
    first_event: str
    second_event: Optional[str]
    max_depth: int
    alternative_found: bool
    alternative_actions: Optional[Tuple[str, ...]]
    expanded_states: int
    state_limit_hit: bool

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "first_event": self.first_event,
            "second_event": self.second_event,
            "max_depth": self.max_depth,
            "alternative_found": self.alternative_found,
            "alternative_actions": list(self.alternative_actions) if self.alternative_actions else None,
            "expanded_states": self.expanded_states,
            "state_limit_hit": self.state_limit_hit,
        }


def _rule_text(rule: Mapping[str, str], roles: RoleNames) -> Optional[str]:
    if "object" in rule and "property" in rule:
        parts = []
        if rule.get("obj_color"):
            parts.append(str(rule["obj_color"]).upper())
        parts.extend((roles.object(rule["object"]), "IS", roles.property(rule["property"])))
        return " ".join(parts)
    if "object1" in rule and "object2" in rule:
        return f"{roles.object(rule['object1'])} IS {roles.object(rule['object2'])}"
    return None


def _token_name(obj, roles: RoleNames) -> Optional[str]:
    if isinstance(obj, RuleColor):
        return f"COLOR[{obj.obj_color.upper()}]"
    if isinstance(obj, RuleObject):
        return roles.object(obj.object)
    if isinstance(obj, RuleProperty):
        return roles.property(obj.property)
    if isinstance(obj, RuleIs):
        return "IS"
    return None


def semantic_snapshot(env, roles: RoleNames) -> SemanticSnapshot:
    rules = frozenset(
        text
        for rule in env.grid._ruleset["_rule_"]
        if (text := _rule_text(rule, roles)) is not None
    )
    positions = {}
    for y in range(env.height):
        for x in range(env.width):
            for obj in env.grid.get(x, y, z="all"):
                if obj is None:
                    continue
                name = _token_name(obj, roles)
                if name is not None:
                    positions.setdefault(name, Counter())[(x, y)] += 1
    return SemanticSnapshot(rules=rules, rule_tokens=positions)


def _moved_rule_tokens(before: SemanticSnapshot, after: SemanticSnapshot) -> Tuple[str, ...]:
    moved = []
    for token_name in sorted(set(before.rule_tokens) | set(after.rule_tokens)):
        old_positions = before.rule_tokens.get(token_name, Counter())
        new_positions = after.rule_tokens.get(token_name, Counter())
        if old_positions == new_positions:
            continue
        moved_count = sum((old_positions - new_positions).values())
        moved.extend([token_name] * max(1, moved_count))
    return tuple(moved)


def _winning_outcomes(env, roles: RoleNames) -> Tuple[str, ...]:
    outcomes = set()
    for y in range(env.height):
        for x in range(env.width):
            objects = [obj for obj in env.grid.get(x, y, z="all") if obj is not None]
            goals = {
                roles.object(obj.type)
                for obj in objects
                if (
                    not isinstance(obj, (RuleColor, RuleObject, RuleProperty, RuleIs))
                    and hasattr(obj, "is_goal")
                    and obj.is_goal()
                )
            }
            if not goals:
                continue
            actors = {
                roles.object(obj.type)
                for obj in objects
                if (
                    not isinstance(obj, (RuleColor, RuleObject, RuleProperty, RuleIs))
                    and hasattr(obj, "is_agent")
                    and obj.is_agent()
                )
            }
            outcomes.update(f"{actor} ON {goal}" for actor in actors for goal in goals)
    return tuple(sorted(outcomes))


def transition_events(
    before: SemanticSnapshot,
    after: SemanticSnapshot,
    child_env,
    roles: RoleNames,
    reward: float,
) -> EventGroup:
    events = [f"PUSH {token}" for token in _moved_rule_tokens(before, after)]
    events.extend(f"REMOVE {rule}" for rule in sorted(before.rules - after.rules))
    events.extend(f"ADD {rule}" for rule in sorted(after.rules - before.rules))
    if reward > 0:
        outcomes = _winning_outcomes(child_env, roles)
        events.append(f"WIN BY {', '.join(outcomes) if outcomes else 'CONTROLLED OBJECT'}")
    return tuple(events)


def append_event(trace: SemanticTrace, event_group: EventGroup) -> SemanticTrace:
    return trace + ((event_group,) if event_group else ())


def rule_strategy_trace(trace: SemanticTrace) -> SemanticTrace:
    """Drop word pushes while retaining rule/control changes and the win target."""

    reduced = []
    for event_group in trace:
        rule_events = tuple(
            event
            for event in event_group
            if event.startswith(("ADD ", "REMOVE ", "WIN BY "))
        )
        if rule_events:
            reduced.append(rule_events)
    return tuple(reduced)


def replay_semantic_trace(env, actions: Sequence[str], roles: RoleNames) -> SemanticTrace:
    state = deepcopy(env)
    trace: SemanticTrace = ()
    for action_name in actions:
        before = semantic_snapshot(state, roles)
        _, reward, _, _ = state.step(getattr(state.actions, action_name))
        after = semantic_snapshot(state, roles)
        trace = append_event(trace, transition_events(before, after, state, roles, reward))
    return trace


def find_win_avoiding_event(
    env,
    event_to_avoid: str,
    *,
    max_depth: int,
    state_limit: int = 250_000,
) -> CounterfactualSearchResult:
    """Find a bounded winning path that never emits one semantic event."""

    roles = RoleNames(env.generation_params)
    initial = deepcopy(env)
    queue = deque([(initial, ())])
    visited = {initial.hash(size=64)}
    expanded_states = 0

    while queue:
        state, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        expanded_states += 1
        before = semantic_snapshot(state, roles)
        for action_name in ACTION_NAMES:
            child = deepcopy(state)
            _, reward, done, _ = child.step(getattr(child.actions, action_name))
            after = semantic_snapshot(child, roles)
            events = transition_events(before, after, child, roles, reward)
            if event_to_avoid in events:
                continue
            child_path = path + (action_name,)
            if reward > 0:
                return CounterfactualSearchResult(
                    kind="avoid_event",
                    first_event=event_to_avoid,
                    second_event=None,
                    max_depth=max_depth,
                    alternative_found=True,
                    alternative_actions=child_path,
                    expanded_states=expanded_states,
                    state_limit_hit=False,
                )
            if done:
                continue
            child_hash = child.hash(size=64)
            if child_hash in visited:
                continue
            visited.add(child_hash)
            if len(visited) > state_limit:
                return CounterfactualSearchResult(
                    kind="avoid_event",
                    first_event=event_to_avoid,
                    second_event=None,
                    max_depth=max_depth,
                    alternative_found=False,
                    alternative_actions=None,
                    expanded_states=expanded_states,
                    state_limit_hit=True,
                )
            queue.append((child, child_path))

    return CounterfactualSearchResult(
        kind="avoid_event",
        first_event=event_to_avoid,
        second_event=None,
        max_depth=max_depth,
        alternative_found=False,
        alternative_actions=None,
        expanded_states=expanded_states,
        state_limit_hit=False,
    )


def find_win_reversing_event_order(
    env,
    first_event: str,
    second_event: str,
    *,
    max_depth: int,
    state_limit: int = 250_000,
) -> CounterfactualSearchResult:
    """Find a bounded win in which ``second_event`` happens before ``first_event``."""

    roles = RoleNames(env.generation_params)
    initial = deepcopy(env)
    # State flags distinguish histories that the grid hash alone cannot: whether
    # the intended first event has happened and whether the order was reversed.
    queue = deque([(initial, (), False, False)])
    visited = {(initial.hash(size=64), False, False)}
    expanded_states = 0

    while queue:
        state, path, first_seen, reversed_order = queue.popleft()
        if len(path) >= max_depth:
            continue
        expanded_states += 1
        before = semantic_snapshot(state, roles)
        for action_name in ACTION_NAMES:
            child = deepcopy(state)
            _, reward, done, _ = child.step(getattr(child.actions, action_name))
            after = semantic_snapshot(child, roles)
            events = transition_events(before, after, child, roles, reward)
            # Events within one transition are atomic, so A and B appearing in
            # the same group do not count as either strict order.
            child_reversed = reversed_order or (
                second_event in events and not first_seen and first_event not in events
            )
            child_first_seen = first_seen or first_event in events
            child_path = path + (action_name,)
            if reward > 0 and child_reversed:
                return CounterfactualSearchResult(
                    kind="reverse_order",
                    first_event=first_event,
                    second_event=second_event,
                    max_depth=max_depth,
                    alternative_found=True,
                    alternative_actions=child_path,
                    expanded_states=expanded_states,
                    state_limit_hit=False,
                )
            if done:
                continue
            key = (child.hash(size=64), child_first_seen, child_reversed)
            if key in visited:
                continue
            visited.add(key)
            if len(visited) > state_limit:
                return CounterfactualSearchResult(
                    kind="reverse_order",
                    first_event=first_event,
                    second_event=second_event,
                    max_depth=max_depth,
                    alternative_found=False,
                    alternative_actions=None,
                    expanded_states=expanded_states,
                    state_limit_hit=True,
                )
            queue.append((child, child_path, child_first_seen, child_reversed))

    return CounterfactualSearchResult(
        kind="reverse_order",
        first_event=first_event,
        second_event=second_event,
        max_depth=max_depth,
        alternative_found=False,
        alternative_actions=None,
        expanded_states=expanded_states,
        state_limit_hit=False,
    )


def run_avoid_event_job(job: Tuple[str, int, str, int, int]) -> dict:
    """Process-pool-friendly wrapper for a bounded event-avoidance search."""

    task, seed, event_to_avoid, max_depth, state_limit = job
    env = baba.make(task)
    env.reset(seed=seed)
    result = find_win_avoiding_event(
        env,
        event_to_avoid,
        max_depth=max_depth,
        state_limit=state_limit,
    ).to_dict()
    result.update({"task": task, "seed": seed, "variant": env.generation_params["variant"]})
    return result


def run_reverse_order_job(job: Tuple[str, int, str, str, int, int]) -> dict:
    """Process-pool-friendly wrapper for a bounded event-order search."""

    task, seed, first_event, second_event, max_depth, state_limit = job
    env = baba.make(task)
    env.reset(seed=seed)
    result = find_win_reversing_event_order(
        env,
        first_event,
        second_event,
        max_depth=max_depth,
        state_limit=state_limit,
    ).to_dict()
    result.update({"task": task, "seed": seed, "variant": env.generation_params["variant"]})
    return result


def analyze_shortest_solutions(env, task: str) -> ShortestSolutionAnalysis:
    """Enumerate every shortest action path and its navigation-free strategy."""

    roles = RoleNames(env.generation_params)
    initial_hash = env.hash(size=64)
    nodes = {
        initial_hash: SearchNode(
            depth=0,
            env=deepcopy(env),
            action_path_count=1,
            trace_examples={(): ()},
        )
    }
    frontier = [initial_hash]
    goal_depth = None
    goal_action_path_count = 0
    goal_trace_examples: Dict[SemanticTrace, Tuple[str, ...]] = {}
    expanded_states = 0

    while frontier and goal_depth is None:
        next_frontier = []
        next_frontier_seen = set()
        for state_hash in frontier:
            node = nodes[state_hash]
            expanded_states += 1
            before = semantic_snapshot(node.env, roles)
            for action_name in ACTION_NAMES:
                child = deepcopy(node.env)
                _, reward, done, _ = child.step(getattr(child.actions, action_name))
                after = semantic_snapshot(child, roles)
                events = transition_events(before, after, child, roles, reward)
                child_depth = node.depth + 1

                if reward > 0:
                    if goal_depth is None:
                        goal_depth = child_depth
                    if child_depth != goal_depth:
                        continue
                    goal_action_path_count += node.action_path_count
                    for trace, example in node.trace_examples.items():
                        goal_trace = append_event(trace, events)
                        goal_trace_examples.setdefault(goal_trace, example + (action_name,))
                    continue

                if done or goal_depth is not None:
                    continue

                child_hash = child.hash(size=64)
                child_node = nodes.get(child_hash)
                if child_node is None:
                    child_node = SearchNode(depth=child_depth, env=child)
                    nodes[child_hash] = child_node
                if child_node.depth != child_depth:
                    continue

                child_node.action_path_count += node.action_path_count
                for trace, example in node.trace_examples.items():
                    child_trace = append_event(trace, events)
                    child_node.trace_examples.setdefault(child_trace, example + (action_name,))
                if child_hash not in next_frontier_seen:
                    next_frontier_seen.add(child_hash)
                    next_frontier.append(child_hash)
        frontier = next_frontier

    if goal_depth is None:
        raise RuntimeError(f"no solution found for {task} seed={env.generation_params.get('seed')}")

    ordered_traces = tuple(sorted(goal_trace_examples, key=repr))
    ordered_rule_traces = tuple(sorted({rule_strategy_trace(trace) for trace in ordered_traces}, key=repr))
    reference_trace = replay_semantic_trace(env, env.reference_solution, roles)
    return ShortestSolutionAnalysis(
        task=task,
        seed=int(env.generation_params["seed"]),
        variant=int(env.generation_params["variant"]),
        reference_length=len(env.reference_solution),
        shortest_length=goal_depth,
        shortest_action_sequence_count=goal_action_path_count,
        semantic_trace_count=len(ordered_traces),
        semantic_traces=ordered_traces,
        semantic_trace_examples=tuple(goal_trace_examples[trace] for trace in ordered_traces),
        rule_strategy_trace_count=len(ordered_rule_traces),
        rule_strategy_traces=ordered_rule_traces,
        reference_trace=reference_trace,
        reference_trace_is_shortest=(
            len(env.reference_solution) == goal_depth
            and reference_trace in goal_trace_examples
        ),
        expanded_states=expanded_states,
    )


def _analyze_task_seed(job: Tuple[str, int]) -> ShortestSolutionAnalysis:
    task, seed = job
    env = baba.make(task)
    env.reset(seed=int(seed))
    return analyze_shortest_solutions(env, task)


def analyze_tasks(
    task_seeds: Mapping[str, Iterable[int]],
    *,
    show_progress: bool = False,
    workers: int = 1,
) -> Tuple[ShortestSolutionAnalysis, ...]:
    jobs = [(task, int(seed)) for task, seeds in task_seeds.items() for seed in seeds]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_analyze_task_seed, jobs)
            analyses = []
            for (task, seed), analysis in zip(jobs, results):
                analyses.append(analysis)
                if show_progress:
                    print(
                        f"finished {task} seed={seed}: shortest={analysis.shortest_length} "
                        f"actions={analysis.shortest_action_sequence_count} "
                        f"word_strategies={analysis.semantic_trace_count} "
                        f"rule_strategies={analysis.rule_strategy_trace_count} "
                        f"states={analysis.expanded_states}",
                        file=sys.stderr,
                        flush=True,
                    )
        return tuple(analyses)

    analyses = []
    for task, seed in jobs:
        if show_progress:
            print(f"analyzing {task} seed={seed}", file=sys.stderr, flush=True)
        analysis = _analyze_task_seed((task, seed))
        analyses.append(analysis)
        if show_progress:
            print(
                f"  shortest={analysis.shortest_length} actions={analysis.shortest_action_sequence_count} "
                f"word_strategies={analysis.semantic_trace_count} "
                f"rule_strategies={analysis.rule_strategy_trace_count} states={analysis.expanded_states}",
                file=sys.stderr,
                flush=True,
            )
    return tuple(analyses)


def _parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        action="append",
        choices=tuple(DEFAULT_TEMPLATE_SEEDS),
        help="analyze one task (repeatable); defaults to all composition tasks",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="override the representative seeds (repeatable)",
    )
    parser.add_argument("--output", type=Path, help="write the complete JSON result to this path")
    parser.add_argument("--workers", type=int, default=1, help="analyze independent seeds in parallel")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    tasks = args.task or list(DEFAULT_TEMPLATE_SEEDS)
    task_seeds = {
        task: tuple(args.seed) if args.seed else DEFAULT_TEMPLATE_SEEDS[task]
        for task in tasks
    }
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    analyses = analyze_tasks(task_seeds, show_progress=True, workers=args.workers)
    payload = {
        "strategy_equivalence": (
            "navigation-only actions are ignored; pushed rule words, active-rule deltas, "
            "and the winning controlled object must match"
        ),
        "analyses": [analysis.to_dict() for analysis in analyses],
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {len(analyses)} analyses to {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
