# This file is part of SliceRL by M. Rossi


import os, argparse, shutil, yaml
from shutil import copyfile
from time import time as tm
import tensorflow as tf

from hyperopt import fmin, tpe, hp, Trials, space_eval
from hyperopt.mongoexp import MongoTrials
import pickle, pprint

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
def run_hyperparameter_scan(search_space, load_data_fn, function):
    """Running a hyperparameter scan using hyperopt.
    """

    print('[+] Performing hyperparameter scan...')
    max_evals = search_space['cluster']['max_evals']
    
    if search_space['cluster']['enable']:
        url = search_space['cluster']['url']
        key = search_space['cluster']['exp_key']
        trials = MongoTrials(url, exp_key=key)
    else:
        env_setup = search_space.get('expurl_env')
        trials = Trials()
    
    generators = load_data_fn(search_space)

    best = fmin(lambda p: function(p, generators), search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    best_setup = space_eval(search_space, best)
    print('\n[+] Best scan setup:')
    pprint.pprint(best_setup)

    log = f"{search_space['output']}/hyperopt_log_{tm()}.pickle"
    with open(log,'wb') as wfp:
        print(f'[+] Saving trials in {log}')
        pickle.dump(trials.trials, wfp)

    # disable scan for final fit
    best_setup['scan'] = False
    return best_setup

#----------------------------------------------------------------------
def load_runcard(runcard_file):
    """ Load runcard from yaml file. """
    with open(runcard_file, 'r') as stream:
        runcard = yaml.load(stream, Loader=yaml.FullLoader)
    runcard['scan'] = False
    for key, value in runcard.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if 'hp.' in str(v):
                    print(k, v, type(v))
                    runcard[key][k] = eval(v)
                    runcard['scan'] = True
        else:
            if 'hp.' in str(value):
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
        '--data',type=str, default=None, dest='data', help='The test set path model.'
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
        from slicerl.build_model import build_and_train_model, inference
        from slicerl.build_dataset import build_dataset_train, build_dataset_test

        # copy runcard to output folder
        copyfile(args.runcard, f'{out}/input-runcard.yaml')

        if setup.get('scan'):
            setup = run_hyperparameter_scan(setup, build_dataset_train, build_and_train_model)

        start = tm()
        print('[+] Training best model:')

        generators = build_dataset_train(setup)
        model = build_and_train_model(setup, generators)
        print(f"[+] done in {tm()-start} s")

        # save the final runcard
        with open(f'{out}/runcard.yaml','w') as f:
            yaml.dump(setup, f, indent=4)
    
    elif args.model:
        folder = args.model.strip('/')
        # loading json card
        setup = load_runcard(f"{folder}/runcard.yaml")
        # loading model
        if args.data:
            setup['test']['fn'] = args.data
    test_generator = build_dataset_test(setup)
    model = inference(setup, test_generator)
        

if __name__=='__main__':
    main()
