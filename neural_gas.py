from adaptive_vector_quantizer import AdaptiveVectorQuantizer

class NeuralGas(AdaptiveVectorQuantizer):
    def __init__(self, max_neurons_n, lifetime='auto', epsilon='auto', lambda_param='auto'):
        super().__init__(max_neurons_n, lifetime)

        self.epsilon = epsilon
        self.lambda_param = lambda_param

    def update(x, iter):
        pass