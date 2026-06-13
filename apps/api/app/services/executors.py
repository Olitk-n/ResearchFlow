from pathlib import Path
from typing import Protocol

from .artifacts import run_experiment_package


class ExperimentExecutor(Protocol):
    name: str

    def run(self, package: Path, timeout_seconds: int) -> tuple[str, dict]: ...


class LocalDockerExecutor:
    name = "docker"

    def run(self, package: Path, timeout_seconds: int) -> tuple[str, dict]:
        return run_experiment_package(package, timeout_seconds)


class DisabledCloudExecutor:
    name = "cloud_disabled"

    def run(self, package: Path, timeout_seconds: int) -> tuple[str, dict]:
        return "blocked", {
            "reason": "云执行器默认关闭，未创建云资源，也未产生云端费用。",
            "experiment_package": str(package),
            "timeout_seconds": timeout_seconds,
            "billable_action": False,
        }


EXECUTORS: dict[str, ExperimentExecutor] = {
    "docker": LocalDockerExecutor(),
    "cloud_disabled": DisabledCloudExecutor(),
}


def execute_experiment(
    executor_name: str,
    package: Path,
    timeout_seconds: int,
) -> tuple[str, dict]:
    return EXECUTORS[executor_name].run(package, timeout_seconds)
