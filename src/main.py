from __future__ import annotations
import hydra
from omegaconf import DictConfig
from test import run_test
from train import run_train, run_train_init
from utils.runtime import configure_logging, start_run_logging


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    LOGGER = configure_logging()

    # Since hydra puts config at top level, we adapt it
    run_dir = start_run_logging(cfg)

    # We can pass an action like mode=train, test, train_init
    mode = cfg.get("mode", "train")
    if mode == "train_init":
        run_train_init(cfg)
    elif mode == "train":
        run_train(cfg)
    elif mode == "test":
        run_test(cfg)
    else:
        LOGGER.error(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
