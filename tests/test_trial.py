import time

from tunekit.core.trial import Trial, TrialStatus


class TestTrial:
    def test_lifecycle(self):
        trial = Trial(params={"lr": 0.01})
        assert trial.status == TrialStatus.PENDING

        trial.start()
        assert trial.status == TrialStatus.RUNNING
        assert trial.start_time is not None

        trial.complete(0.95, {"epoch": 10})
        assert trial.status == TrialStatus.COMPLETED
        assert trial.score == 0.95
        assert trial.metrics["epoch"] == 10
        assert trial.duration is not None
        assert trial.duration >= 0

    def test_fail(self):
        trial = Trial(params={"lr": 0.1})
        trial.start()
        trial.fail("CUDA OOM")
        assert trial.status == TrialStatus.FAILED
        assert trial.metrics["error"] == "CUDA OOM"

    def test_prune(self):
        trial = Trial(params={"lr": 0.5})
        trial.start()
        trial.prune()
        assert trial.status == TrialStatus.PRUNED

    def test_duration_none_before_complete(self):
        trial = Trial(params={})
        assert trial.duration is None
