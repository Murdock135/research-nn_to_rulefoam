from config import Config
import sklearn.datasets as skdatasets
import matplotlib.pyplot as plt

# local imports
from utils.data_io import make_clusters

if __name__ == "__main__":
    # create synthetic dataset
    X = make_clusters()

    # scatter plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(X[:, 0], X[:, 1], cmap='viridis', edgecolor='k')

    plt.show()
