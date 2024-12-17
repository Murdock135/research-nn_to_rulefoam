from adaptive_vector_quantizer import AdaptiveVectorQuantizer

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
        super().__init__()

    def update(x, iter):
        pass
    