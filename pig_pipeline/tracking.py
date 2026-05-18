from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


class RunTracker:
    def __init__(self, config: dict[str, Any], run_name: str, group: str | None = None):
        wb_config = config.get("wandb", {})
        self.enabled = bool(wb_config.get("enabled", False)) and wandb is not None
        self.run = None
        if self.enabled:
            self.run = wandb.init(
                project=wb_config.get("project", "pig-posture"),
                entity=wb_config.get("entity"),
                tags=wb_config.get("tags", []),
                mode=wb_config.get("mode", "online"),
                name=run_name,
                group=group,
                config=config,
                settings=wandb.Settings(_disable_stats=True)
            )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def log_file(self, key: str, path: str | Path) -> None:
        if self.run is None:
            return

        file_path = Path(path)
        if not file_path.exists():
            return

        artifact = wandb.Artifact(name=f"{self.run.id}-{key}", type="output")
        artifact.add_file(str(file_path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
