import pickle
import pprint
from time import time as tm
from hyperopt import fmin, tpe, hp, Trials, space_eval
from hyperopt.mongoexp import MongoTrials


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

    wrap_fn = lambda x: function(x, load_data_fn(x))

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

    log = search_space["output"] / f"hopt/hyperopt_log_{tm()}.pickle"
    with open(log, "wb") as wfp:
        print(f"[+] Saving trials in {log}")
        pickle.dump(trials.trials, wfp)

    # disable scan for final fit
    best_setup["scan"] = False
    from slicerl.plot_hyperopt import plot_hyperopt

    plot_hyperopt(log)
    return best_setup
