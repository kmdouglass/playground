# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "PySide6",
# ]
# ///
import matplotlib.pyplot as plt
import numpy as np


def main(p=0.01, num_trials=10000, repeats=100000, num_bins=100) -> None:
    """Probabilistic read noise model.
    
    This models read noise as a sum of a large number of independent, discrete
    Bernoulli trials, each of which has a small probability of generating an extra
    electron.
    
    """
    rvs = np.random.binomial(1, p, size=(num_trials, repeats))
    results = rvs.sum(axis=0)
    
    # Plot the histogram over repeats
    plt.hist(results, bins=num_bins)
    plt.show()

if __name__ == "__main__":
    plt.switch_backend("QtAgg")
    main(p=0.01, num_bins=100)
