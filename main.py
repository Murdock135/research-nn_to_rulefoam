from config import Config
import sklearn.datasets as skdatasets
import matplotlib.pyplot as plt

# local imports
from utils.data_io import make_clusters
from growing_neural_gas import GrowingNeuralGas

if __name__ == "__main__":
    # create synthetic dataset
    X, y = make_clusters()

    # scatter plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(X[:, 0], X[:, 1], c = y, cmap='viridis', edgecolor='k')

    vector_quantizer = GrowingNeuralGas()
    vector_quantizer.run(X, max_iter=X.shape[0])


    plt.show()
