import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from tqdm import tqdm
import utils.plot_utils as pu
from growing_neural_gas import GrowingNeuralGas
from utils.data_io import create_save_path, SyntheticDataset
import imageio
import os


# # Plotting setup
# fig, ax = plt.subplots()
# color_dict = pu.NG_colors
# cmap = ListedColormap([color_dict["data"], "#ffd1df", "salmon", "red"])
# norm = BoundaryNorm(boundaries=[0, 1, 2, 3, 4], ncolors=cmap.N)

# # Create colorbar
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# tick_pos = [0.5, 1.5, 2.5, 3.5]
# cbar = fig.colorbar(sm, ax=ax, ticks=tick_pos)
# cbar.set_ticklabels(list(range(epochs + 1)))
# cbar.set_label("Number of times sampled")


def plot_gng(data, neurons, current_sample, iter, connection_matrix=None, save_path=None):
    plt.figure()

    # plot data, neurons, current sample
    plt.scatter(data[:,0], data[:,1], marker='o', label='data')
    plt.scatter(neurons[:,0], neurons[:,1], marker='x', label='neuron')
    plt.scatter(current_sample[0], current_sample[1], label='current sample')

    # plot connections
    if connection_matrix is not None:
        n1s, n2s = np.nonzero(np.triu(connection_matrix, k=1))
        for n1, n2 in zip(n1s, n2s):
            neuron_1 = neurons[n1]
            neuron_2 = neurons[n2]
            x_coords = [neuron_1[0], neuron_2[0]]
            y_coords = [neuron_1[1], neuron_2[1]]
            plt.plot(
                x_coords,
                y_coords,
                linestyle="-",
                c='k'
            )

    plt.title(f"Iteration {iter}")
    plt.legend()
    if save_path is not None:
            plt.savefig(f"{save_path}/{iter}.png")
    plt.close()

def create_gif(results_dir):
    images = []
    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith(".png"):
            images.append(imageio.imread(os.path.join(results_dir, filename)))
    imageio.mimsave(os.path.join(results_dir, 'gng_evolution.gif'), images)


# Training loop
# def train_epoch():

# for epoch in tqdm(range(epochs), desc="Epoch"):
#     random_seq = quantizer.get_random_sequence(data)

#     if quantizer.sampling_without_replacement:
#         new_cmap_colors = [cmap(epoch), cmap(epoch + 1)]
#         new_cmap = ListedColormap(new_cmap_colors)

#     for i in tqdm(range(quantizer.max_iter)):
#         choice = random_seq[i]
#         x = quantizer.data[choice]
#         quantizer.sample_counts[choice] += 1
#         quantizer.update(i, x)

#         if i % plot_interval == 0 or i == quantizer.max_iter - 1:
#             plot_GNG(
#                 data=quantizer.data,
#                 neurons=quantizer.neurons,
#                 sample_counts=quantizer.sample_counts,
#                 current_sample=x,
#                 connection_matrix=quantizer.connection_matrix,
#                 epoch=epoch,
#                 iter=i,
#                 cmap=new_cmap,
#             )
#             pu.save_fig(results_dir, epoch, i)

#     # Create GIF
#     pu._create_gif(results_dir, epoch)

print("Training completed")

if __name__ == "__main__":
    # Load data
    data_path = r"data/dataset1.png"
    extractor = SyntheticDataset(data_path)
    df = extractor.extract_data()
    data = extractor.get_coordinates_as_numpy(df)

    # Create save path
    results_dir = r"results/"
    results_dir = create_save_path(results_dir, "GNG")

    # Create model
    epochs = 3
    plot_interval = 100
    quantizer = GrowingNeuralGas(data=data,
                                 results_dir=results_dir,)

    neurons_n = 2
    quantizer.neurons = quantizer.create_neurons(neurons_n)
    quantizer.connection_matrix = np.zeros((neurons_n, neurons_n))
    random_seq = quantizer.get_random_sequence(data)

    for i in tqdm(range(quantizer.max_iter)):
         choice = random_seq[i]
         x = data[choice]
         quantizer.update(i, x)

         if i % plot_interval == 0 or i == quantizer.max_iter-1:
              plot_gng(data, quantizer.neurons, x, i, quantizer.connection_matrix, results_dir)
    
    # create gif
    create_gif(results_dir)
