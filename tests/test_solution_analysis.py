import unittest

import baba

from baba.solution_analysis import (
    analyze_shortest_solutions,
    find_win_avoiding_event,
)


class SolutionAnalysisTest(unittest.TestCase):
    def analyze(self, task, seed):
        env = baba.make(task)
        env.reset(seed=seed)
        return analyze_shortest_solutions(env, task)

    def test_navigation_paths_collapse_to_one_rule_strategy(self):
        analysis = self.analyze("env/composition-07-one-token-fork", 3)

        self.assertEqual(analysis.shortest_length, 6)
        self.assertEqual(analysis.shortest_action_sequence_count, 6)
        self.assertEqual(analysis.semantic_trace_count, 1)
        self.assertEqual(analysis.rule_strategy_trace_count, 1)
        self.assertTrue(analysis.reference_trace_is_shortest)
        self.assertIn(
            "WIN BY BABA ON TARGET",
            tuple(event for group in analysis.semantic_traces[0] for event in group),
        )

    def test_word_push_variations_keep_the_same_rule_strategy(self):
        analysis = self.analyze("env/composition-01-target-lock", 6)

        self.assertEqual(analysis.shortest_length, 6)
        self.assertEqual(analysis.shortest_action_sequence_count, 9)
        self.assertEqual(analysis.semantic_trace_count, 7)
        self.assertEqual(analysis.rule_strategy_trace_count, 1)
        self.assertTrue(analysis.reference_trace_is_shortest)

    def test_bounded_search_can_rule_out_avoiding_a_key_event(self):
        task = "env/composition-07-one-token-fork"
        env = baba.make(task)
        env.reset(seed=3)

        result = find_win_avoiding_event(
            env,
            "ADD TARGET IS WIN",
            max_depth=12,
            state_limit=250_000,
        )

        self.assertFalse(result.alternative_found)
        self.assertFalse(result.state_limit_hit)
        self.assertGreater(result.expanded_states, 0)


if __name__ == "__main__":
    unittest.main()
