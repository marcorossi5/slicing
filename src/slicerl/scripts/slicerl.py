# This file is part of SliceRL by M. Rossi

"""
    slicerl.py: the entry point for slicerl.
"""
from slicerl.read_data import Events, save_Event_list_to_file, load_Events_from_file
from slicerl.Event import Event
from slicerl.models import build_and_train_model, load_runcard, load_environment
from slicerl.diagnostics import inference, make_plots, load_and_dump_plots
from slicerl.Slicer import ContinuousSlicer
from slicerl.keras_to_cpp import keras_to_cpp, check_model
from hyperopt import fmin, tpe, hp, Trials, space_eval
from hyperopt.mongoexp import MongoTrials
from time import time as tm
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
    TODO: better import/export for the best model, wait to DQNAgentSlicerl
    """

    print('[+] Performing hyperparameter scan...')
    max_evals = search_space['cluster']['max_evals']
    if search_space['cluster']['enable']:
        url = search_space['cluster']['url']
        key = search_space['cluster']['exp_key']
        trials = MongoTrials(url, exp_key=key)
        slicerl_env = None
    else:
        env_setup = search_space.get('slicerl_env')
        slicerl_env = load_environment(env_setup)
        trials = Trials()

    best = fmin(lambda p: build_and_train_model(p, slicerl_env), search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    best_setup = space_eval(search_space, best)
    print('\n[+] Best scan setup:')
    pprint.pprint(best_setup)

    log = '%s/hyperopt_log_{}.pickle'.format(tm()) % search_space['output']
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
    for key, value in runcard.get('slicerl_env').items():
        if 'hp.' in str(value):
            runcard['slicerl_env'][key] = eval(value)
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
def safe_inference_and_plots(slicer, fnin, fnres, plotdir, loaddir, nev, visualize=False, gifname=None):
    """
    Make inference from dataset at fnin and save results at fnres. Output
    diagnostic plots in plotdir and plot data in loaddir. Perform checks trying
    to skip already stored computations. This function never overwrites files.

    try to make plotdir-loaddir.
    if they already exist and contain data --> load_and_dump
    else do nothing
    if they did not exist
        if inference data file already exist --> load Events and make plots
        else --> make inference
        make plots
    

    Parameters
    ----------
        fnin      : str, input dataset file name
        fnres     : str, inference results file name
        plotdir   : str, folder name to store diagnostic plots
        loaddir   : str, folder name to store diagnostic plots data (to tweak plots only)
        nev       : int, number of event to load
        visualize : bool, wether to create gif animation of actor scores
        gifname   : str, filename to save gif
    """
    try:
        makedir(plotdir)
        makedir(loaddir)
    except:
        if len(os.listdir(loaddir)) > 0:
            print(f'[+] Plotting on previously collected results: {loaddir} already exists')
            load_and_dump_plots(plotdir, loaddir)
            # TODO: we cannot plot_plane_view without loading the events
        else:
            print(f'[+] Ignoring plots: {loaddir} already exists, but is empty')
    else:
        print(f'[+] Creating test plots in {plotdir}')
        # if already tested slicer, load results and make plots
        # otherwise do inference first
        if os.path.isfile(fnres):
            print('[+] Json file found: loading from saved test data')
            events = load_Events_from_file(fnres, nev, num_lines=7)
        else:
            events = load_Events_from_file(fnin, nev)
            if visualize:
                if gifname is None:
                    raise ValueError('Invalid gifname')
                assert gifname.split(os.extsep)[-1] == 'gif', f"got {gifname}, gif filename must have .gif extension"
            events = inference(slicer, events, visualize, gifname)
            save_Event_list_to_file(events, fnres)
        make_plots(events, plotdir)

#----------------------------------------------------------------------
def main():
    """Parsing command line arguments"""
    start = tm()
    # read command line arguments
    parser = argparse.ArgumentParser(description='Train an ML slicer.')
    parser.add_argument('runcard', action='store', nargs='?', default=None,
                        help='A json file with the setup.')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='The input model.')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='The output folder.')
    parser.add_argument('--plot',action='store_true',dest='plot')
    parser.add_argument('--visualize',action='store_true',dest='visualize')
    parser.add_argument('--force', '-f', action='store_true',dest='force',
                        help='Overwrite existing files if present')
    parser.add_argument('--cpp',action='store_true',dest='cpp')
    parser.add_argument('--data', type=str, default=None, dest='data',
                        help='Data on which to apply the slicer.')
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

        # slicerl common environment setup
        if setup.get('scan'):
            rl_agent_setup = run_hyperparameter_scan(setup)
        else:
            # create the DQN agent and train it.
            rl_agent_setup = setup

        print('[+] Training best model:')
        ddpg = build_and_train_model(rl_agent_setup)

        # save the final runcard
        with open(f'{out}/runcard.json','w') as f:
            json.dump(rl_agent_setup, f, indent=4)

        fnres = '%s/test_predictions.csv.gz' % setup['output']

        print('[+] Done with training, now testing on sample set')
        if os.path.exists(fnres):
            os.remove(fnres)

        slicer = ddpg.slicer()
        events = load_Events_from_file(setup['test']['fn'], args.nev)
        gifname = '%s/test_predictions_actor_scores.gif' % setup['output']
        events = inference(slicer, events, visualize=args.visualize, gifname=gifname)
        save_Event_list_to_file(events, fnres)

        # define the folder where to do the plotting/cpp conversation
        folder = setup['output']

    elif args.model:
        # raise ValueError('Loading of model has not been implemented yet')
        folder = args.model.strip('/')
        # loading json card
        setup = load_runcard('%s/runcard.json' % folder)
        rl_agent_setup = setup
        # loading slicer
        slicer = ContinuousSlicer()
        modelwgts_fn = '%s/weights.h5' % folder
        modeljson_fn = '%s/model.json' % folder
        slicer.load_with_json(modeljson_fn, modelwgts_fn)

    # if requested, add plotting
    if args.plot:
        fnin    = setup['test']['fn']
        plotdir = '%s/plots' % folder
        loaddir = '%s/results' % plotdir
        fnres   = '%s/test_predictions.csv.gz' % setup['output']
        gifname = '%s/test_predictions_actor_scores.gif' % setup['output']
        safe_inference_and_plots(slicer, fnin, fnres, plotdir, loaddir, args.nev, args.visualize, gifname)

    # if a data set was given as input, produce plots from it
    # always check if inference data is already there
    if args.data:
        fnin = os.path.basename(args.data).split(os.extsep)[0]
        plotdir='%s/%s' % (folder, fnin)
        loaddir = '%s/results' % plotdir
        fnres = '%s/%s_sliced.csv.gz' % (folder, fnin)
        gifname = '%s/%s_sliced_actor_scores.gif' % (folder, fnin)
        safe_inference_and_plots(slicer, args.data, fnres, plotdir, loaddir, args.nev, args.visualize, gifname)

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
            arch_dic=ast.literal_eval(slicer.model.to_json()
                                      .replace('true','True')
                                      .replace('null','None'))
            keras_to_cpp(slicer.model, arch_dic['config']['layers'], cpp_fn)
    print(f"Program done in {tm()-start} s")
