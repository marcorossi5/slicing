from . import make_plots_hc, make_plots_cm, make_plots_common


def make_plots(setup, generator):
    """Wrapper function for plots."""
    folder = setup["output"] / "plots"
    # common plots
    make_plots_common.make_plots(generator, folder)

    # architecture specific plots
    modeltype = setup["model"]["net_type"]
    if modeltype == "CM":
        make_plots_cm.make_plots(generator, folder)
    elif modeltype == "HC":
        make_plots_hc.make_plots(generator, folder)
    else:
        raise NotImplementedError(f"Architecture not implemented, found {modeltype}")
