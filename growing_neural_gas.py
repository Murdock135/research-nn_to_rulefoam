from adaptive_vector_quantizer import AdaptiveVectorQuantizer
import numpy as np

class GrowingNeuralGas(AdaptiveVectorQuantizer):
    def __init__(self,
                 init_neurons_num=2,
                 max_neurons_num=1000,
                 eps_b=0.2,
                 eps_n=0.006,
                 lambda_param=100,
                 alpha=0.5,
                 decay=0.995
    ):
        super().__init__(max_neurons_n=max_neurons_num)
        self.init_neurons_num = init_neurons_num
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.lambda_param = lambda_param
        self.alpha = alpha
        self.decay = decay

    def update(self, x, neurons, iter):
        # calculate distances between neurons and data and then find the 2 closest neurons
        distances = np.linalg.norm(x - neurons, axis=1)
        s1_idx, s2_idx = np.argsort(distances)[:2]

        # find neighbors of the winner, increase ages and then move them
        neighbor_indices = np.nonzero(self.age_matrix[s1_idx])[0]
        for neighbor_idx in neighbor_indices:
            self.increase_age(s1_idx, neighbor_idx)
            neurons[neighbor_idx] += self.eps_n * (x - neurons[neighbor_idx])
        
        # move the best neuron
