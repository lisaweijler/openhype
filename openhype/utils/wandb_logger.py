from dataclasses import asdict, dataclass, field
import torch
import wandb
import os
from typing import Dict, List


@dataclass
class WandBConfig:
    project_name: str = "to be set"
    team: str = "to be set"
    notes: str = None
    # tags: List[str] = field(default_factory=List)
    group: str = None
    enabled: bool = True

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


class WandBLogger:

    def __init__(
        self,
        config: WandBConfig,
        model: torch.nn.modules = None,
        run_name: str = None,
        run_config: Dict = None,
    ) -> None:

        if config is None:
            self.enabled = False
        else:
            self.enabled = config.enabled

        if self.enabled:
            wandb.init(
                entity=config.team,
                project=config.project_name,
                group=config.group,
                notes=config.notes,
                config=run_config.dict(),
            )
            if run_name is None:
                wandb.run.name = wandb.run.id
            else:
                wandb.run.name = run_name

            if model is not None:
                self.watch(model)

    def watch(self, model, log_freq: int = 1):
        wandb.watch(model, log="all", log_freq=log_freq)

    def log(self, log_dict: dict, commit=True, step=None):
        if self.enabled:
            if step:
                wandb.log(log_dict, commit=commit, step=step)
            else:
                wandb.log(log_dict, commit=commit)

    def save_bestmodel(self, state):
        if self.enabled:
            torch.save(state, os.path.join(wandb.run.dir, "model_best.pth"))

    def finish(self):
        if self.enabled:
            wandb.finish()
