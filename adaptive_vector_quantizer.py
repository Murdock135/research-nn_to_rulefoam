import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Note on the constuction of this class:
# - Only kept hyperparameters as the class attributes (with the exception of the age matrix)
class AdaptiveVectorQuantizer:
    def __init__(self, max_neurons_n, lifetime='auto'):
        self.max_neurons_n = max_neurons_n
        self.lifetime = lifetime
        self.age_matrix = None

    def run(self, X, epochs=1, max_iter=1000, replace=False, plot_n=100, results_dir=None):
        # Check data sampling logic
        if replace == False:
            assert max_iter <= X.shape[0], f"AssertionError: max_iter ({max_iter}) exceeds number of samples ({X.shape[0]}). Cannot sample without replacement."

        # Create neurons
        neurons = self.create_neurons(X)

        # Create age matrix
        n = neurons.shape[0]
        self.age_matrix = np.zeros((n, n))

        # Initialize sample counts
        sample_counts = np.zeros(X.shape[0])

        # Set plot interval
        plot_interval = int(max_iter/plot_n)

        for epoch in range(epochs):
            # Shuffle data
            rng = np.random.default_rng(0)
            X_shuffled = rng.choice(X, size=max_iter, replace=replace)

            # Run core algorithm
            for i in tqdm(range(max_iter)):
                # Get sample
                x = X_shuffled[i]

                # Increase the sample count for x
                mask = np.all(X == x, axis=1)
                sample_counts[mask] += 1

                # Run update algorithm
                self.update(x, neurons, i)

                if i % plot_interval == 0 | i == max_iter-1:
                    self._plot(X, neurons, x, sample_counts, epoch, i)
    
    # TODO
    def update(self, x, neurons, iter):
        pass
        
    def create_neurons(self, X, n=2, dist='uniform'):
        """Create initial neurons for the network.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        n : int, default=2
            Number of initial neurons to create.
        dist : {'uniform', 'normal'}, default='uniform'
            Distribution to sample initial neuron positions from:
            - 'uniform': Sample uniformly between min and max values of X
            - 'normal': Sample from normal distribution with mean and std of X

        Returns
        -------
        ndarray of shape (n, n_features)
            Initial neuron positions.

        
        """
        dim = X.shape[1]
        min_values = np.amin(X, axis=0)
        max_values = np.amax(X, axis=0)
        rng = np.random.default_rng(0)

        if dist.lower() == "uniform":
            return rng.uniform(low=min_values, high=max_values, size=(n, dim))
        
        elif dist.lower() == "normal":
            mean_arr = np.mean(self.data, axis=0)
            std_arr = np.std(self.data, axis=0)

            return rng.normal(mean_arr, std_arr, size=(n, dim))
        else:
            print(f"Distribution {dist} not recognized. Use either 'uniform' or 'normal'")
    
    def increase_age(self, r_index, c_index):
        if self.connection_matrix[r_index, c_index] < self.lifetime:
            self.connection_matrix[r_index, c_index] += 1
            self.connection_matrix[c_index, r_index] += 1

    def remove_old_connections(self):
        """Remove connections older than the specified lifetime and delete lonely neurons."""
        old_connections = self.connection_matrix > self.lifetime
        self.connection_matrix[old_connections] = 0

    def _plot(self, X, neurons, current_x, sample_counts, epoch=None, iter=None, save_path=None):
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], color=sample_counts, label='Data')
        ax.scatter(neurons[:, 0], neurons[:, 1], label='Neurons')
        ax.scatter(current_x[0], current_x[1], label='Current Sample')
        self._plot_connections(fig, neurons, self.age_matrix)

        # Name plot
        ax.title(f"Epoch {epoch if epoch is not None else 1}\n\
                  Iteration {iter if iter is not None else 1}.")
        ax.xlabel("X")
        ax.ylabel("y")
        ax.legend()

        # save figure
        # TODO: Provide directory. Create a separate file to create directory paths
        if save_path is not None:
            fig.savefig()

    def _plot_connections(self, ax, neurons, age_matrix):
        head_nodes_indxs, tail_node_idxs = np.nonzero(np.triu(age_matrix, k=1))

        for head_node_idx, tail_node_idx in zip(head_nodes_indxs, tail_node_idxs):
            head_node = neurons[head_node_idx]
            tail_node = neurons[tail_node_idx]

            X = [head_node, tail_node]
            ax.plot(X[:, 0], X[:, 1])
    