# This file is part of SliceRL by M. Rossi
import os, argparse, shutil, yaml
from pathlib import Path
from shutil import copyfile
from time import time as tm
import tensorflow as tf

from hyperopt import fmin, tpe, hp, Trials, space_eval
from hyperopt.mongoexp import MongoTrials
import pickle, pprint

# ======================================================================
def config_tf(setup):
    os.environ["CUDA_VISIBLE_DEVICES"] = setup.get("gpu")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# ======================================================================
def load_runcard(runcard_file):
    """Load runcard from yaml file."""
    with open(runcard_file, "r") as stream:
        runcard = yaml.load(stream, Loader=yaml.FullLoader)
    runcard["scan"] = False
    for key, value in runcard.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if "hp." in str(v):
                    runcard[key][k] = eval(v)
                    runcard["scan"] = True
        else:
            if "hp." in str(value):
                runcard[key][k] = eval(value)
                runcard["scan"] = True
    return runcard


# ======================================================================
def modify_runcard(setup):
    setup.update({"output": Path(setup["output"])})
    setup["train"].update({"dataset_dir": Path(setup["train"]["dataset_dir"])})
    setup["test"].update({"dataset_dir": Path(setup["test"]["dataset_dir"])})


# ======================================================================
def check_dataset_directory(
    dataset_dir, should_load_dataset=False, should_save_dataset=False
):
    """
    If should_load_dataset is true, checks if dataset directory exists and
    raises error if not. If should_save_dataset is true, checks if dataset
    directory exists and creates it. Raises error if parent folder does not exist.

    Parameters
    ----------
        - dataset_dir: Path
        - should_load_dataset: bool
        - should_save_dataset: bool

    Raises
    ------
        - FileNotFoundError if directory does not exist
    """
    if should_load_dataset and not dataset_dir.exists():
        raise FileNotFoundError(f"no such file or directory: {dataset_dir}")

    if should_save_dataset and not dataset_dir.exists():
        dataset_dir.mkdir()


# ======================================================================
def run_hyperparameter_scan(search_space, load_data_fn, function):
    """Running a hyperparameter scan using hyperopt."""

    print("[+] Performing hyperparameter scan...")
    max_evals = search_space["cluster"]["max_evals"]

    if search_space["cluster"]["enable"]:
        url = search_space["cluster"]["url"]
        key = search_space["cluster"]["exp_key"]
        trials = MongoTrials(url, exp_key=key)
    else:
        env_setup = search_space.get("expurl_env")
        trials = Trials()

    def wrap_fn(setup):
        generators = load_data_fn(setup)
        return function(setup, generators)

    best = fmin(
        wrap_fn,
        search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )

    best_setup = space_eval(search_space, best)
    print("\n[+] Best scan setup:")
    pprint.pprint(best_setup)

    log = search_space["output"].joinpath(f"hopt/hyperopt_log_{tm()}.pickle")
    with open(log, "wb") as wfp:
        print(f"[+] Saving trials in {log}")
        pickle.dump(trials.trials, wfp)

    # disable scan for final fit
    best_setup["scan"] = False
    from slicerl.plot_hyperopt import plot_hyperopt

    plot_hyperopt(log)
    return best_setup


# ======================================================================
def main():
    ss = tm()
    parser = argparse.ArgumentParser(
        """
    Example script to train RandLA-Net for slicing
    """
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        dest="data",
        help="The test set path model.",
    )
    parser.add_argument(
        "--save_dataset",
        help="Save training dataset in folder specified by runcard",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_dataset_test",
        help="Save training dataset in folder specified by runcard",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--load_dataset",
        help="Load training dataset from folder specified by runcard",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--load_dataset_test",
        help="Load test dataset from folder specified by runcard",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--debug",
        help="Run TensorFlow eagerly",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--force",
        "-f",
        help="Overwrite existing files if present",
        action="store_true",
        dest="force",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="The input model."
    )
    parser.add_argument("--output", "-o", help="Output folder", type=Path, default=None)
    parser.add_argument(
        "runcard",
        action="store",
        nargs="?",
        type=Path,
        default=None,
        help="A yaml file with the setup.",
    )
    parser.add_argument(
        "--no_graphics",
        help="DO not display figures via GUI",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    # check that input is coherent
    if (not args.model and not args.runcard) or (args.model and args.runcard):
        raise ValueError("Invalid options: requires either input runcard or model.")
    elif args.runcard and not args.runcard.is_file():
        raise ValueError("Invalid runcard: not a file.")
    if args.force:
        print("WARNING: Running with --force option will overwrite existing model")
    if (args.save_dataset and args.load_dataset) or (
        args.save_dataset_test and args.load_dataset_test
    ):
        raise ValueError("Invalid options: requires either save or load dataset.")

    setup = {}
    if args.debug:
        print("[+] Run all tf functions eagerly")
        tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()
        setup["debug"] = True

    if args.runcard:
        # load yaml
        setup.update(load_runcard(args.runcard))
        config_tf(setup)

        from slicerl.build_dataset import build_dataset_train
        from slicerl.build_model import build_and_train_model

        # create output folder
        out = args.runcard.suffix[0]
        if args.output is not None:
            out = args.output
        from slicerl.tools import makedir

        try:
            makedir(out)
            makedir(out.joinpath("plots"))
            if setup.get("scan"):
                makedir(out.joinpath("hopt"))
        except Exception as error:
            if args.force:
                print(f"WARNING: Overwriting {out} with new model")
                shutil.rmtree(out)
                makedir(out)
                makedir(out.joinpath("plots"))
                if setup.get("scan"):
                    makedir(out.joinpath("hopt"))
            else:
                print(error)
                print('Delete or run with "--force" to overwrite.')
                exit(-1)
        setup["output"] = out
        modify_runcard(setup)
        check_dataset_directory(
            setup["train"]["dataset_dir"],
            should_load_dataset=args.load_dataset,
            should_save_dataset=args.save_dataset,
        )

        # copy runcard to output folder
        copyfile(args.runcard, out.joinpath("input-runcard.yaml"))

        if setup.get("scan"):
            setup = run_hyperparameter_scan(
                setup, build_dataset_train, build_and_train_model
            )

        start = tm()
        print("[+] Training best model:")

        generators = build_dataset_train(
            setup,
            should_load_dataset=args.load_dataset,
            should_save_dataset=args.save_dataset,
        )
        build_and_train_model(setup, generators)
        print(f"[+] done in {tm()-start} s")

        # save the final runcard
        with open(out.joinpath("runcard.yaml"), "w") as f:
            # yaml is not able to save the Path objects
            # TODO: overload the yaml class
            setup["output"] = setup["output"].as_posix()
            setup["train"]["dataset_dir"] = setup["train"]["dataset_dir"].as_posix()
            setup["test"]["dataset_dir"] = setup["test"]["dataset_dir"].as_posix()
            yaml.dump(setup, f, indent=4)
            modify_runcard(setup)

    elif args.model:
        folder = Path(args.model.strip("/"))
        # loading json card
        setup = load_runcard(folder.joinpath("runcard.yaml"))
        modify_runcard(setup)

        config_tf(setup)

        # loading model
        if args.data:
            setup["test"]["fn"] = args.data

    check_dataset_directory(
        setup["test"]["dataset_dir"],
        should_load_dataset=args.load_dataset_test,
        should_save_dataset=args.save_dataset_test,
    )

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
