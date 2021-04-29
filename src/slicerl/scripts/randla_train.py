# This file is part of SliceRL by M. Rossiimport os, argparse, shutil, yaml
import os, argparse, shutil, yaml
from shutil import copyfile
from time import time as tm
import tensorflow as tf
from slicerl.build_model import build_and_train_model

def config_tf(setup):
    os.environ["CUDA_VISIBLE_DEVICES"] = setup.get('gpu')
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

#----------------------------------------------------------------------
def makedir(folder):
    """Create directory."""
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        raise Exception('Output folder %s already exists.' % folder)

#----------------------------------------------------------------------
def load_runcard(runcard_file):
    """ Load runcard from yaml file. """
    with open(runcard_file, 'r') as stream:
        runcard = yaml.load(stream, Loader=yaml.FullLoader)
    runcard['scan'] = False
    for key, value in runcard.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if ('hp.' in str(value)) or ('None' in str(value)):
                    runcard[key][k] = eval(value)
                    runcard['scan'] = True
        else:
            if ('hp.' in str(value)) or ('None' in str(value)):
               runcard[key][k] = eval(value)
               runcard['scan'] = True
    return runcard

#======================================================================
def main():
    parser = argparse.ArgumentParser(
        """
    Example script to train RandLA-Net for slicing
    """
    )
    parser.add_argument(
        'runcard', action='store', nargs='?', default=None, help='A yaml file with the setup.'
    )
    parser.add_argument(
        '--model', '-m', type=str, default=None, help='The input model.'
    )
    parser.add_argument(
        "--output", "-o", help="Output folder", type=str, default=None
    )
    parser.add_argument(
        '--force', '-f', help='Overwrite existing files if present', action='store_true', dest='force',                        
    )
    parser.add_argument(
        "--debug", help="Run TensorFlow eagerly", action="store_true", default=False
    )
    args = parser.parse_args()
    # check that input is coherent
    if (not args.model and not args.runcard) or (args.model and args.runcard):
        raise ValueError('Invalid options: requires either input runcard or model.')
    elif args.runcard and not os.path.isfile(args.runcard):
        raise ValueError('Invalid runcard: not a file.')
    elif args.model and not (args.plot or args.cpp or args.data):
        raise ValueError('Invalid options: no actions requested.')
    if args.force:
        print('WARNING: Running with --force option will overwrite existing model')

    setup = {}
    if args.debug:
        print("[+] Run all tf functions eagerly")
        tf.config.run_functions_eagerly(True)
        setup['debug'] = True

    if args.runcard:
        # load yaml
        setup.update( load_runcard(args.runcard) )
        config_tf(setup)

        # create output folder
        base = os.path.basename(args.runcard)
        out = os.path.splitext(base)[0]
        if args.output is not None:
            out = args.output
        try:
            makedir(out)
        except Exception as error:
            if args.force:
                print(f'WARNING: Overwriting {out} with new model')
                shutil.rmtree(out)
                makedir(out)
            else:
                print(error)
                print('Delete or run with "--force" to overwrite.')
                exit(-1)
        setup['output'] = out

        # copy runcard to output folder
        copyfile(args.runcard, f'{out}/input-runcard.yaml')

        # here it goes the training !
        print('[+] Training best model:')
        start = tm()
        model = build_and_train_model(setup)
        print(f"[+] done in {tm()-start} s")


        # save the final runcard
        with open(f'{out}/runcard.yaml','w') as f:
            yaml.dump(setup, f, indent=4)

if __name__=='__main__':
    main()
