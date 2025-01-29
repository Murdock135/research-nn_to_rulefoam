import os

class Config:
    # paths
    RESULTS_DIR = "results/"
    NG_RESULTS_DIR = os.path.join(RESULTS_DIR, "neural_gas")
    GNG_RESULTS_DIR = os.path.join(RESULTS_DIR, "growing_neural_gas")

    # experiment attributes
    seed = 50
