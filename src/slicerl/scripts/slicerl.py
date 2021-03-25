# This file is part of SliceRL by M. Rossi

"""
    expurl.py: the entry point for expurl.
"""
from slicerl.read_data import Events, save_Event_list_to_file, load_Events_from_file
from slicerl.Event import Event
from slicerl.models import build_and_train_model, load_runcard, load_environment
from slicerl.diagnostics import inference, make_plots, load_and_dump_plots
from slicerl.Slicer import ContinuousSlicer
from slicerl.keras_to_cpp import keras_to_cpp, check_model
from hyperopt import fmin, tpe, hp, Trials, space_eval
from hyperopt.mongoexp import MongoTrials
import fastjet as fj
from time import time
from shutil import copyfile
from copy import deepcopy
import os, argparse, pickle, pprint, json, gzip, ast, shutil
#import cProfile

os.environ['CUDA_VISIBLE_DEVICES'] = ""

#----------------------------------------------------------------------
def run_hyperparameter_scan(search_space):
    """Running a hyperparameter scan using hyperopt.
    TODO: implement cross-validation, e.g. k-fold, or randomized cross-validation.
    TODO: use test data as hyper. optimization goal.
    TODO: better import/export for the best model, wait to DQNAgentExpurl
    """

    print('[+] Performing hyperparameter scan...')
    max_evals = search_space['cluster']['max_evals']
    if search_space['cluster']['enable']:
        url = search_space['cluster']['url']
        key = search_space['cluster']['exp_key']
        trials = MongoTrials(url, exp_key=key)
        expurl_env = None
    else:
        env_setup = search_space.get('expurl_env')
        expurl_env = load_environment(env_setup)
        trials = Trials()

    best = fmin(lambda p: build_and_train_model(p, expurl_env), search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    best_setup = space_eval(search_space, best)
    print('\n[+] Best scan setup:')
    pprint.pprint(best_setup)

    log = '%s/hyperopt_log_{}.pickle'.format(time()) % search_space['output']
    with open(log,'wb') as wfp:
        print(f'[+] Saving trials in {log}')
        pickle.dump(trials.trials, wfp)

    # disable scan for final fit
    best_setup['scan'] = False
    return best_setup

#----------------------------------------------------------------------
def load_json(runcard_file):
    """Loads json, execute python expressions, and sets
    scan flags accordingly to the syntax.
    """
    runcard = load_runcard(runcard_file)
    runcard['scan'] = False
    for key, value in runcard.get('expurl_env').items():
        if 'hp.' in str(value):
            runcard['expurl_env'][key] = eval(value)
            runcard['scan'] = True
    for key, value in runcard.get('rl_agent').items():
        if 'hp.' in str(value):
            runcard['rl_agent'][key] = eval(value)
            runcard['scan'] = True
    return runcard

#----------------------------------------------------------------------
def makedir(folder):
    """Create directory."""
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        raise Exception('Output folder %s already exists.' % folder)

#----------------------------------------------------------------------
def safe_inference_and_plots(subtractor, fnin, fnres, plotdir, loaddir, nev, R):
    """
    Make inference from dataset at fnin and save results at fnres. Output
    diagnostic plots in plotdir and plot data in loaddir. Perform checks trying
    to skip already stored computations. This function never overwrites files.

    Parameters
    ----------
        fnin: str
            input dataset file name
        fnres: str
            inference results file name
        plotdir: str
            folder name to store diagnostic plots
        loaddir: str
            folder name to store diagnostic plots data (to tweak plots only)
        nev: int
            number of event to load
        R: float
            radius parameter of jet clustering algorithm
    """
    try:
        makedir(plotdir)
        makedir(loaddir)
    except:
        if len(os.listdir(loaddir)) > 0:
            print(f'[+] Plotting on previously collected results: {loaddir} already exists')
            load_and_dump_plots(plotdir, loaddir)
        else:
            print(f'[+] Ignoring plots: {loaddir} already exists, but is empty')
    else:
        print(f'[+] Creating test plots in {plotdir}')
        # if already tested subtractor, load results and make plots
        # otherwise do inference first
        if os.path.isfile(fnres):
            print('[+] Json file found: loading from saved test data')
            events = load_Jets_from_file(fnres, nev, R, True)
        else:
            events = load_Jets_from_file(fnin, nev, R, True)
            events = inference(subtractor, events)
            save_Event_list_to_file(events, fnres)
        make_plots(events, fj.antikt_algorithm, R, plotdir)


#----------------------------------------------------------------------
def main():
    """Parsing command line arguments"""
    # read command line arguments
    parser = argparse.ArgumentParser(description='Train an ML subtractor.')
    parser.add_argument('runcard', action='store', nargs='?', default=None,
                        help='A json file with the setup.')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='The input model.')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='The output folder.')
    parser.add_argument('--plot',action='store_true',dest='plot')
    parser.add_argument('--force', '-f', action='store_true',dest='force',
                        help='Overwrite existing files if present')
    parser.add_argument('--cpp',action='store_true',dest='cpp')
    parser.add_argument('--data', type=str, default=None, dest='data',
                        help='Data on which to apply the subtractor.')
    parser.add_argument('--nev', '-n', type=float, default=-1,
                        help='Number of events.')
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

    if args.runcard:
        # load json
        setup = load_json(args.runcard)

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
        copyfile(args.runcard, f'{out}/input-runcard.json')

        # expurl common environment setup
        if setup.get('scan'):
            rl_agent_setup = run_hyperparameter_scan(setup)
        else:
            # create the DQN agent and train it.
            rl_agent_setup = setup

        print('[+] Training best model:')
        dqn = build_and_train_model(rl_agent_setup)

        # save the final runcard
        with open(f'{out}/runcard.json','w') as f:
            json.dump(rl_agent_setup, f, indent=4)

        fnres = '%s/test_predictions.json.gz' % setup['output']

        print('[+] Done with training, now testing on sample set')
        if os.path.exists(fnres):
            os.remove(fnres)

        subtractor = dqn.subtractor()
        events = load_Jets_from_file(setup['test']['fn'], args.nev, setup['expurl_env']['jet_R'], True)
        events = inference(subtractor, events)
        save_Event_list_to_file(events, fnres)

        # define the folder where to do the plotting/cpp conversation
        folder = setup['output']

    elif args.model:
        # raise ValueError('Loading of model has not been implemented yet')
        folder = args.model.strip('/')
        # loading json card
        setup = load_runcard('%s/runcard.json' % folder)
        rl_agent_setup = setup
        # loading subtractor
        if setup['expurl_env']['discrete']:
            subtractor = DiscreteSubtractor()
        else:
            raise ValueError('ContinuousSubtractor was not implemented yet')
            # subtractor = ContinuousSubtractor()
        modelwgts_fn = '%s/weights.h5' % folder
        modeljson_fn = '%s/model.json' % folder
        subtractor.load_with_json(modeljson_fn, modelwgts_fn)

    # if requested, add plotting
    if args.plot:
        fnin    = setup['test']['fn']
        plotdir = '%s/plots' % folder
        loaddir = '%s/results' % plotdir
        fnres   = '%s/test_predictions.json.gz' % setup['output']
        safe_inference_and_plots(subtractor, fnin, fnres, plotdir, loaddir, args.nev, setup['expurl_env']['jet_R'])

    # if a data set was given as input, produce plots from it
    # always check if inference data is already there
    if args.data:
        fnin = os.path.basename(args.data).split(os.extsep)[0]
        plotdir='%s/%s' % (folder, fnres)
        loaddir = '%s/results' % plotdir
        fnres = '%s/%s_sb.json.gz' % (plotdir, fnin)
        safe_inference_and_plots(subtractor, fnin, fnres, plotdir, loaddir, args.nev, setup['expurl_env']['jet_R'])

    # if requested, add cpp output
    if args.cpp:
        check_model(rl_agent_setup['rl_agent'])
        cppdir = '%s/cpp' % folder
        try:
            makedir(cppdir)
        except:
            print(f'[+] Ignoring cpp instruction: {cppdir} already exists')
        else:
            print(f'[+] Adding cpp model in {cppdir}')
            cpp_fn = '%s/model.nnet' % cppdir
            arch_dic=ast.literal_eval(subtractor.model.to_json()
                                      .replace('true','True')
                                      .replace('null','None'))
            keras_to_cpp(subtractor.model, arch_dic['config']['layers'], cpp_fn)
