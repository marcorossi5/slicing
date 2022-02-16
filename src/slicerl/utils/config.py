"""
    This module reads the command line options and initializes the directory
    structure.
"""
import shutil
from slicerl.utils.configflow import config_tf
from slicerl.utils.utils import (
    get_cmd_args,
    check_cmd_args,
    initialize_output_folder,
    check_dataset_directory,
    load_runcard,
    modify_runcard,
)


def config_init():
    args = get_cmd_args()
    check_cmd_args(args)
    
    setup = {"debug": args.debug}
    if args.runcard:
        setup.update(load_runcard(args.runcard))
        initialize_output_folder(args.output, args.force, setup.get("scan"))
        setup["output"] = args.output
        shutil.copyfile(args.runcard, args.output / "input-runcard.yaml")
    elif args.model:
        setup = load_runcard(args.model / "runcard.yaml")
    else:
        raise ValueError("Check inputs, you shouldn't be here !")

    modify_runcard(setup)
    config_tf(setup)

    # check dataset directory structure
    check_dataset_directory(
        setup["train"]["dataset_dir"],
        should_load_dataset=args.load_dataset,
        should_save_dataset=args.save_dataset,
    )
    check_dataset_directory(
        setup["test"]["dataset_dir"],
        should_load_dataset=args.load_dataset_test,
        should_save_dataset=args.save_dataset_test,
    )
    return args
