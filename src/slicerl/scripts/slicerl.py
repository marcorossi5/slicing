# This file is part of SliceRL by M. Rossi
from time import time as tm
from slicerl.utils.config import config_init
from slicerl.utils.utils import save_runcard


def main():
    ss = tm()
    args = config_init()

    if args.runcard:
        from slicerl.build_dataset import (
            build_dataset_train,
            build_dataset_from_np,
            save_dataset_np,
        )
        from slicerl.build_model import build_and_train_model

        if setup.get("scan"):
            from slicerl.hopt.hopt import run_hyperparameter_scan

            setup = run_hyperparameter_scan(
                setup, build_dataset_train, build_and_train_model
            )

        start = tm()
        print("[+] Training best model:")
        just_train_path = setup["dataset_dir"].parent / "training"
        if args.just_train:
            generators = build_dataset_from_np(setup, just_train_path)
        else:
            generators = build_dataset_train(
                setup,
                should_load_dataset=args.load_dataset,
                should_save_dataset=args.save_dataset,
            )
            if args.save_train_np:
                save_dataset_np(generators, just_train_path)
        build_and_train_model(setup, generators)
        save_runcard(setup["output"] / "runcard.yaml", setup)
        print(f"[+] done in {tm()-start} s")

    from slicerl.build_dataset import build_dataset_test
    from slicerl.build_model import inference

    test_generator = build_dataset_test(
        setup,
        should_load_dataset=args.load_dataset_test,
        should_save_dataset=args.save_dataset_test,
    )
    inference(setup, test_generator, no_graphics=args.no_graphics)
    print(f"[+] Program done in {tm()-ss} s")


if __name__ == "__main__":
    main()
