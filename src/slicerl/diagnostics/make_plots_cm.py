import matplotlib.pyplot as plt
from .make_plots_common import plot_histogram

def make_plots(generator, folder):
    """
    Additional plots for CM-Net diagnostics.

    Parameters
    ----------
        - generator: EventDataset, the test generator
        - folder: Path, the output folder
    """
    # add functions here
    hist_true = [trg.flatten() for trg in generator.targets]
    hist_pred = [pred.flatten() for pred in generator.y_pred.all_y_pred]
    plot_histogram(hist_true, hist_pred, folder)

