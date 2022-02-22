# This file is part of SliceRL by M. Rossi
import logging
from time import time as tm
from slicerl import PACKAGE
from slicerl.utils.config import config_init
from slicerl.utils.utils import save_runcard

logger = logging.getLogger(PACKAGE)


def main():
    """SliceRL entry point."""
    ss = tm()
    args, setup = config_init()
    from slicerl.build_model import build_and_train_model, inference
    from slicerl.dataloading.build_dataset import build_dataset
    from slicerl.dataloading.cmnet_dataset import save_dataset_np
    from slicerl.hopt.hopt import run_hyperparameter_scan

    if args.runcard:
        if setup.get("scan"):
            setup = run_hyperparameter_scan(setup, build_dataset, build_and_train_model)

        start = tm()
        logger.info("Training best model")
        from_np_path = setup["train"]["dataset_dir"].parent / "training"
        if args.just_train:
            logger.info(f"Importing dataset from {from_np_path}")
            generators = build_dataset(setup, from_np_path=from_np_path)
        else:
            generators = build_dataset(
                setup,
                is_training=True,
                should_load_dataset=args.load_dataset,
                should_save_dataset=args.save_dataset,
            )
            if args.save_train_np:
                logger.info(f"Saving dataset to {from_np_path}")
                save_dataset_np(generators, from_np_path)
                exit()
        build_and_train_model(setup, generators, args.seed)
        save_runcard(setup["output"] / "runcard.yaml", setup)
        logger.info(f"Training done in {tm()-start} s")

    test_generator = build_dataset(
        setup,
        is_training=False,
        should_load_dataset=args.load_dataset_test,
        should_save_dataset=args.save_dataset_test,
    )
    inference(setup, test_generator, no_graphics=args.no_graphics)
    logger.info(f"[+] Program done in {tm()-ss} s")


if __name__ == "__main__":
    main()
