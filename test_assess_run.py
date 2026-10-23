import unittest

try:
    from .assess_run import evaluate_saved_samples
except ImportError:
    from assess_run import evaluate_saved_samples


def sample(time, grips, passed=True, seam=4):
    return {"time_s": time, "grips_active": grips,
            "threading_passed": passed, "active_seam_count": seam}


class ReleaseAssessmentTest(unittest.TestCase):
    def test_partial_trial_never_passes_without_release(self):
        result = evaluate_saved_samples([sample(0, [False, False]),
                                         sample(2, [True, True]), sample(6, [True, True])])
        self.assertFalse(result["sampled_task_success"])

    def test_initially_open_grippers_are_not_a_release(self):
        result = evaluate_saved_samples([sample(0, [False, False]), sample(2, [False, False])])
        self.assertFalse(result["sampled_task_success"])

    def test_short_hold_fails(self):
        result = evaluate_saved_samples([sample(0, [True, True]),
                                         sample(1, [False, False]), sample(1.8, [False, False])])
        self.assertFalse(result["sampled_task_success"])

    def test_complete_sampled_hold_passes(self):
        result = evaluate_saved_samples([sample(0, [True, True]),
                                         sample(1, [False, False]), sample(2.1, [False, False])])
        self.assertTrue(result["sampled_task_success"])
        self.assertAlmostEqual(result["final_qualifying_hold_duration_s"], 1.1)

    def test_lost_sleeve_or_seam_resets_hold(self):
        for failed in (sample(1.8, [False, False], passed=False), sample(1.8, [False, False], seam=3)):
            result = evaluate_saved_samples([sample(0, [True, True]), sample(1, [False, False]),
                                             failed, sample(2, [False, False]), sample(2.5, [False, False])])
            self.assertFalse(result["sampled_task_success"])
            self.assertAlmostEqual(result["final_qualifying_hold_duration_s"], .5)

    def test_earlier_numerical_explosion_invalidates_later_apparent_hold(self):
        exploded = sample(.5, [True, True])
        exploded["numerical_state_plausible"] = False
        result = evaluate_saved_samples([sample(0, [True, True]), exploded,
                                         sample(1, [False, False]), sample(2.1, [False, False])])
        self.assertFalse(result["sampled_task_success"])
        self.assertTrue(result["numerical_stretch_failure_observed"])


if __name__ == "__main__":
    unittest.main()
