import pickle
import matplotlib.pyplot as plt
import pprint
import pandas as pd
import numpy as np
import seaborn as sns


def build_dataframe(trials, bestid):
    data = {}
    data["iteration"] = [t["tid"] for t in trials]
    data["loss"] = [t["result"]["loss"] for t in trials]

    for p, k in enumerate(trials[0]["misc"]["vals"].keys()):
        data[k] = [t["misc"]["vals"][k][0] for t in trials]

    df = pd.DataFrame(data)
    bestdf = df[df["iteration"] == bestid["tid"]]
    return df, bestdf


# ----------------------------------------------------------------------
def plot_scans(df, bestdf, trials, bestid, file):
    print("plotting scan results...")
    # plot loss
    nplots = len(trials[0]["misc"]["vals"].keys()) + 1
    f, axs = plt.subplots(1, nplots, sharey=True, figsize=(50, 10))

    axs[0].scatter(df.get("iteration"), df.get("loss"))
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Loss")
    axs[0].set_yscale("log")
    axs[0].scatter(bestdf.get("iteration"), bestdf.get("loss"))

    # plot features
    for p, k in enumerate(trials[0]["misc"]["vals"].keys()):

        # use scatter for variables with many possible values, violin plot otherwise
        if k in ("K", "dropout", "lr"):
            axs[p + 1].scatter(df.get(k), df.get("loss"))
            if k in "learning_rate":
                axs[p + 1].set_xscale("log")
                axs[p + 1].set_xlim([1e-5, 1])
        else:
            sns.violinplot(
                x=df.get(k),
                y=df.get("loss"),
                ax=axs[p + 1],
                palette="Set2",
                cut=0.0,
            )
            sns.stripplot(
                x=df.get(k),
                y=df.get("loss"),
                ax=axs[p + 1],
                color="gray",
                alpha=0.4,
            )
        axs[p + 1].set_xlabel(k)
        axs[p + 1].scatter(bestdf.get(k), bestdf.get("loss"), color="orange")

    plt.savefig(f"{file}", bbox_inches="tight")


# ----------------------------------------------------------------------
def plot_correlations(df, file):
    print("plotting correlations...")
    plt.figure(figsize=(20, 20))
    sns.heatmap(
        df.corr(),
        mask=np.zeros_like(df.corr(), dtype=np.bool),
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        square=True,
        vmax=1,
        vmin=-1,
        annot=True,
        fmt=".2f",
    )
    plt.savefig(f"{file}", bbox_inches="tight")


# ----------------------------------------------------------------------
def plot_pairs(df, file):
    print("plotting pairs")
    plt.figure(figsize=(50, 50))
    sns.pairplot(df)
    plt.savefig(f"{file}", bbox_inches="tight")


# ----------------------------------------------------------------------
def plot_hyperopt(trials_fname):
    """
    Params
    ------
        - trials_fname: Path, trials file name
    """
    """Load trials and generate plots"""
    with open(trials_fname, "rb") as f:
        input_trials = pickle.load(f)

    print("Filtering bad scans...")
    trials = []
    best = 10000
    bestid = -1
    for t in input_trials:
        if t["state"] == 2:
            trials.append(t)
            if t["result"]["loss"] < best:
                best = t["result"]["loss"]
                bestid = t
    print(f"Number of good trials {len(trials)}")
    pprint.pprint(bestid)

    # compute dataframe
    df, bestdf = build_dataframe(trials, bestid)

    # plot scans
    plot_scans(
        df, bestdf, trials, bestid, trials_fname.with_name("hopt_scan.png")
    )

    # plot correlation matrix
    plot_correlations(df, trials_fname.with_name("hopt_corr.png"))

    # plot pairs
    plot_pairs(df, trials_fname.with_name("hopt_pairs.png"))
