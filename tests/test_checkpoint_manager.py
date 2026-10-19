from pathlib import Path

import pytest

from src.safety.checkpoint_manager import CheckpointManager


class State:
    def __init__(self, value=0):
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


def save(manager: CheckpointManager, name: str, step: int, safe: bool, domain: float):
    state = State(step).state_dict()
    return manager.save(
        name, optimizer_step=step, control_step=step, capability_score=0.99,
        domain_validation_score=domain, safety_verified=safe, pid_state=state,
        safety_state=state, trainer_state=state, optimizer_state=state,
    )


def test_latest_safe_best_feasible_and_restore(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    save(manager, "safe-one", 1, True, 0.8)
    save(manager, "unsafe", 2, False, 1.0)
    save(manager, "safe-three", 3, True, 0.9)
    assert manager.latest_safe.checkpoint_id == "safe-three"
    assert manager.best_feasible.checkpoint_id == "safe-three"
    request = manager.request_rollback("hard violation")
    assert request.target_checkpoint_id == "safe-three"
    targets = [State() for _ in range(4)]
    manager.restore(
        "safe-three", pid=targets[0], safety=targets[1],
        trainer=targets[2], optimizer=targets[3],
    )
    assert [target.value for target in targets] == [3, 3, 3, 3]
    assert manager.rollback_request is None


def test_unsafe_checkpoint_cannot_be_rollback_target(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    save(manager, "unsafe", 1, False, 1.0)
    with pytest.raises(PermissionError):
        manager.restore(
            "unsafe", pid=State(), safety=State(), trainer=State(), optimizer=State()
        )


def test_artifacts_are_written_before_checkpoint_is_published(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    state = State().state_dict()

    def write_artifact(directory: Path) -> None:
        (directory / "weights.marker").write_text("complete", encoding="utf-8")

    record = manager.save(
        "atomic", optimizer_step=1, control_step=1, capability_score=1.0,
        domain_validation_score=None, safety_verified=True, pid_state=state,
        safety_state=state, trainer_state=state, optimizer_state=state,
        artifact_writer=write_artifact,
    )
    assert (Path(record.path) / "weights.marker").read_text(encoding="utf-8") == "complete"
