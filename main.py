from config import Config

import sklearn.datasets as skdatasets

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # create synthetic dataset
    X, y = skdatasets.make_classification(
        n_samples = 350,
        n_classes=4,
        n_informative=2,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state = 0
    )

    # scatter plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')

    plt.show()
