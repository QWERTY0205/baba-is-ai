import os
import unittest
from collections import deque
from copy import deepcopy

import baba


TASKS = (
    "env/composition-01-target-lock",
    "env/composition-02-stop-to-win",
    "env/composition-03-reuse-is",
    "env/composition-04-reuse-win",
    "env/composition-05-cross-rule",
    "env/composition-06-color-scope",
    "env/composition-07-one-token-fork",
    "env/composition-08-control-handoff",
    "env/composition-09-ordered-assembly",
    "env/composition-10-control-relay",
)

EXPECTED_SIZES = {
    "env/composition-01-target-lock": (8, 8),
    "env/composition-02-stop-to-win": (8, 8),
    "env/composition-03-reuse-is": (13, 9),
    "env/composition-04-reuse-win": (8, 8),
    "env/composition-05-cross-rule": (13, 9),
    "env/composition-06-color-scope": (8, 8),
    "env/composition-07-one-token-fork": (8, 8),
    "env/composition-08-control-handoff": (8, 8),
    "env/composition-09-ordered-assembly": (8, 8),
    "env/composition-10-control-relay": (13, 9),
}

# These seeds cover every geometric template. Searches for the larger or
# longer tasks stay opt-in because proving minimality exhaustively is slower.
FAST_SHORTEST_SEEDS = {
    "env/composition-01-target-lock": (0, 2, 3, 6),
    "env/composition-02-stop-to-win": (0, 2, 3, 6),
    "env/composition-04-reuse-win": (0, 3, 7, 18),
    "env/composition-06-color-scope": (0, 1, 3, 7),
    "env/composition-07-one-token-fork": (0, 2, 3, 6),
    "env/composition-08-control-handoff": (0, 1, 3, 7),
}
SLOW_SHORTEST_SEEDS = {
    "env/composition-03-reuse-is": (0, 3, 7, 18),
    "env/composition-05-cross-rule": (0, 3, 7, 18),
    "env/composition-09-ordered-assembly": (0, 2, 3, 6),
    "env/composition-10-control-relay": (0, 1, 3, 7),
}

ACTION_NAMES = ("up", "right", "down", "left")


def active_rules(env):
    return {
        (rule["object"], rule["property"])
        for rule in env.grid._ruleset["_rule_"]
        if "object" in rule and "property" in rule
    }


def expected_rules(task, params):
    target = f"f{params['target']}" if "target" in params else None
    if task == "env/composition-01-target-lock":
        return {("baba", "is_agent"), (target, "is_goal"), (target, "is_stop")}
    if task == "env/composition-02-stop-to-win":
        return {("baba", "is_agent"), (target, "is_stop")}
    if task == "env/composition-03-reuse-is":
        return {("baba", "is_agent"), ("fwall", "is_stop")}
    if task == "env/composition-04-reuse-win":
        return {("baba", "is_agent"), (f"f{params['wrong_target']}", "is_goal")}
    if task == "env/composition-05-cross-rule":
        return {("baba", "is_agent"), ("fwall", "is_stop"), (target, "is_goal")}
    if task == "env/composition-06-color-scope":
        return {
            ("baba", "is_agent"),
            (f"f{params['blocker']}", "is_stop"),
            (f"f{params['goal']}", "is_goal"),
        }
    if task == "env/composition-07-one-token-fork":
        return {("baba", "is_agent")}
    if task == "env/composition-08-control-handoff":
        return {("baba", "is_agent"), (f"f{params['goal']}", "is_goal")}
    if task == "env/composition-09-ordered-assembly":
        return {("baba", "is_agent"), (f"f{params['distractor']}", "is_stop")}
    if task == "env/composition-10-control-relay":
        return {("baba", "is_agent"), (f"f{params['goal']}", "is_goal")}
    raise ValueError(f"unknown composition task: {task}")


def object_position(env, obj_type, color=None):
    for y in range(env.height):
        for x in range(env.width):
            cell = env.grid.get(x, y)
            if cell is not None and cell.type == obj_type and (color is None or cell.color == color):
                return x, y
    raise AssertionError(f"{color or ''} {obj_type} not found")


def replay_reference_solution(env):
    reward = 0
    done = False
    for name in env.reference_solution:
        _, reward, done, _ = env.step(getattr(env.actions, name))
    return reward, done


def shortest_solution(env, depth_limit):
    queue = deque([(env, ())])
    visited = {env.hash()}

    while queue:
        state, path = queue.popleft()
        if len(path) >= depth_limit:
            continue

        for name in ACTION_NAMES:
            child = deepcopy(state)
            _, reward, done, _ = child.step(getattr(child.actions, name))
            child_path = path + (name,)
            if reward > 0:
                return child_path
            if done:
                continue

            state_hash = child.hash()
            if state_hash not in visited:
                visited.add(state_hash)
                queue.append((child, child_path))

    return None


class RuleCompositionEnvTest(unittest.TestCase):
    def test_generated_sizes_match_existing_benchmark_layouts(self):
        for task, expected_size in EXPECTED_SIZES.items():
            with self.subTest(task=task):
                env = baba.make(task)
                env.reset(seed=0)
                self.assertEqual((env.width, env.height), expected_size)

    def test_same_seed_reproduces_the_same_level(self):
        for task in TASKS:
            for seed in (0, 1, 17, 42):
                with self.subTest(task=task, seed=seed):
                    first = baba.make(task)
                    first.reset(seed=seed)
                    second = baba.make(task)
                    second.reset(seed=seed)
                    self.assertEqual(first.hash(), second.hash())
                    self.assertEqual(first.generation_params, second.generation_params)
                    self.assertEqual(first.reference_solution, second.reference_solution)

    def test_legacy_seed_then_reset_is_reproducible(self):
        for task in TASKS:
            with self.subTest(task=task):
                legacy = baba.make(task)
                legacy.seed(42)
                legacy.reset()
                modern = baba.make(task)
                modern.reset(seed=42)
                self.assertEqual(legacy.hash(), modern.hash())

    def test_different_seeds_generate_diverse_levels(self):
        for task in TASKS:
            signatures = set()
            variants = set()
            for seed in range(32):
                env = baba.make(task)
                env.reset(seed=seed)
                signatures.add(env.hash())
                variants.add(env.generation_params["variant"])

            with self.subTest(task=task):
                self.assertGreaterEqual(len(signatures), 28)
                self.assertEqual(variants, {0, 1, 2, 3})

    def test_initial_rules_match_generation_metadata(self):
        for task in TASKS:
            for seed in range(8):
                with self.subTest(task=task, seed=seed):
                    env = baba.make(task)
                    env.reset(seed=seed)
                    self.assertEqual(active_rules(env), expected_rules(task, env.generation_params))

    def test_reference_solutions_win_across_seeds(self):
        for task in TASKS:
            for seed in range(20):
                with self.subTest(task=task, seed=seed):
                    env = baba.make(task)
                    env.reset(seed=seed)
                    reward, done = replay_reference_solution(env)
                    self.assertTrue(done)
                    self.assertGreater(reward, 0)

    def test_control_handoff_moves_control_from_baba_to_target(self):
        task = "env/composition-08-control-handoff"
        for seed in (0, 1, 3, 7):
            with self.subTest(seed=seed):
                env = baba.make(task)
                env.reset(seed=seed)
                params = env.generation_params
                target_type = f"f{params['target']}"
                goal_type = f"f{params['goal']}"
                target_color = params["target_color"]
                twin_color = params["target_twin_color"]
                target_position = object_position(env, target_type, target_color)
                twin_position = object_position(env, target_type, twin_color)

                env.step(getattr(env.actions, env.reference_solution[0]))
                baba_position = object_position(env, "baba")
                rule_specs = {
                    (rule["object"], rule["property"], rule.get("obj_color"))
                    for rule in env.grid._ruleset["_rule_"]
                    if "object" in rule and "property" in rule
                }
                self.assertEqual(rule_specs, {
                    (target_type, "is_agent", target_color),
                    (goal_type, "is_goal", None),
                })
                self.assertTrue(env.grid.get(*target_position).is_agent())
                self.assertFalse(env.grid.get(*twin_position).is_agent())

                env.step(getattr(env.actions, env.reference_solution[1]))
                self.assertEqual(object_position(env, "baba"), baba_position)
                self.assertNotEqual(object_position(env, target_type, target_color), target_position)
                self.assertEqual(object_position(env, target_type, twin_color), twin_position)

    def test_color_scope_narrows_stop_to_one_color(self):
        task = "env/composition-06-color-scope"
        for seed in (0, 1, 3, 7):
            with self.subTest(seed=seed):
                env = baba.make(task)
                env.reset(seed=seed)
                params = env.generation_params
                blocker_type = f"f{params['blocker']}"
                blocker_pos = object_position(env, blocker_type, params["blocker_color"])
                twin_pos = object_position(env, blocker_type, params["scope_color"])

                env.step(getattr(env.actions, env.reference_solution[0]))
                scoped_rule = next(
                    rule
                    for rule in env.grid._ruleset["_rule_"]
                    if rule.get("object") == blocker_type and rule.get("property") == "is_stop"
                )
                self.assertEqual(scoped_rule.get("obj_color"), params["scope_color"])
                self.assertFalse(env.grid.get(*blocker_pos).is_stop())
                self.assertTrue(env.grid.get(*twin_pos).is_stop())

    def test_one_token_fork_activates_only_the_selected_target(self):
        task = "env/composition-07-one-token-fork"
        for seed in (0, 2, 3, 6):
            with self.subTest(seed=seed):
                env = baba.make(task)
                env.reset(seed=seed)
                for action in env.reference_solution[:2]:
                    env.step(getattr(env.actions, action))
                self.assertIn((f"f{env.generation_params['target']}", "is_goal"), active_rules(env))
                self.assertNotIn((f"f{env.generation_params['decoy']}", "is_goal"), active_rules(env))

    def test_ordered_assembly_prepares_win_before_sealing_the_passage(self):
        task = "env/composition-09-ordered-assembly"
        for seed in (0, 2, 3, 6):
            with self.subTest(seed=seed):
                env = baba.make(task)
                env.reset(seed=seed)
                distractor_type = f"f{env.generation_params['distractor']}"
                distractor_pos = object_position(env, distractor_type)
                self.assertTrue(env.grid.get(*distractor_pos).is_stop())
                self.assertNotIn(
                    "fwall",
                    {cell.type for cell in env.grid if cell is not None},
                )
                env.step(getattr(env.actions, env.reference_solution[0]))
                self.assertEqual(active_rules(env), expected_rules(task, env.generation_params))
                for action in env.reference_solution[1:-2]:
                    env.step(getattr(env.actions, action))
                self.assertEqual(
                    active_rules(env),
                    {("baba", "is_agent"), (f"f{env.generation_params['target']}", "is_goal")},
                )

    def test_control_relay_restores_baba_then_removes_target_control(self):
        task = "env/composition-10-control-relay"
        for seed in (0, 1, 3, 7):
            with self.subTest(seed=seed):
                env = baba.make(task)
                env.reset(seed=seed)
                params = env.generation_params
                target_type = f"f{params['target']}"
                goal_type = f"f{params['goal']}"

                env.step(getattr(env.actions, env.reference_solution[0]))
                self.assertEqual(
                    {
                        (rule["object"], rule["property"], rule.get("obj_color"))
                        for rule in env.grid._ruleset["_rule_"]
                        if "object" in rule and "property" in rule
                    },
                    {(target_type, "is_agent", params["target_color"]), (goal_type, "is_goal", None)},
                )

                for action in env.reference_solution[1:8]:
                    env.step(getattr(env.actions, action))
                self.assertEqual(
                    active_rules(env),
                    {(target_type, "is_agent"), ("baba", "is_agent"), (goal_type, "is_goal")},
                )

                env.step(getattr(env.actions, env.reference_solution[8]))
                self.assertEqual(active_rules(env), {("baba", "is_agent"), (goal_type, "is_goal")})

    def test_single_room_templates_have_shortest_reference_solutions(self):
        for task, seeds in FAST_SHORTEST_SEEDS.items():
            for seed in seeds:
                with self.subTest(task=task, seed=seed):
                    env = baba.make(task)
                    env.reset(seed=seed)
                    solution = shortest_solution(env, len(env.reference_solution))
                    self.assertIsNotNone(solution)
                    self.assertEqual(len(solution), len(env.reference_solution))

    @unittest.skipUnless(
        os.environ.get("BABA_RUN_SLOW_TESTS") == "1",
        "set BABA_RUN_SLOW_TESTS=1 to exhaustively check larger/longer templates",
    )
    def test_large_templates_have_shortest_reference_solutions(self):
        for task, seeds in SLOW_SHORTEST_SEEDS.items():
            for seed in seeds:
                with self.subTest(task=task, seed=seed):
                    env = baba.make(task)
                    env.reset(seed=seed)
                    solution = shortest_solution(env, len(env.reference_solution))
                    self.assertIsNotNone(solution)
                    self.assertEqual(len(solution), len(env.reference_solution))


if __name__ == "__main__":
    unittest.main()
