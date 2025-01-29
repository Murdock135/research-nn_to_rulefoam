# Standard library imports
import os
from datetime import datetime

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from PIL import Image

# Local imports
from utils.data_colors import COLOR_MAP

seed = 0
class SyntheticDataset:
    def __init__(self, image_path, color_map=COLOR_MAP, tolerance=10):
        self.color_map = color_map
        self.tolerance = tolerance

    def _is_color_close(self, color1, color2):
        return all(abs(c1 - c2) <= self.tolerance for c1, c2 in zip(color1, color2))

    def _find_closest_class(self, pixel_color):
        for known_color, label in self.color_map.items():
            if self._is_color_close(known_color, pixel_color):
                return label
        return None

    def extract_data(self) -> pd.DataFrame:
        # Open the image
        img = Image.open(self.image_path)

        # Convert the image to RGBA (if it's not already in that format)
        img = img.convert("RGBA")

        # Get the size of the image
        width, height = img.size

        # Prepare a list to hold coordinates and color of colored points
        data = []

        # Loop through each pixel in the image
        for y in tqdm(range(height), desc="Extracting data"):
            for x in range(width):
                pixel_color = img.getpixel((x, y))[
                    :3
                ]  # Get RGB value, ignoring alpha channel
                label = self._find_closest_class(pixel_color)
                if label:
                    data.append(((x, y), pixel_color, label))

        df = pd.DataFrame(data, columns=["coordinates", "pixel value", "class"])
        return df

    def normalize_coordinates(self, df) -> np.ndarray:
        coordinates = self.get_coordinates_as_numpy(df)
        tqdm.pandas(desc="Normalizing coordinates")
        scaler = MinMaxScaler()
        normalized_coordinates = scaler.fit_transform(coordinates)
        return normalized_coordinates

    def get_coordinates_as_numpy(self, df):
        return np.vstack(df["coordinates"].values)
    
    def plot_data(self, df, normalized=False):
        """Plots the data points with color coding based on class."""
        plt.figure(figsize=(8, 8))

        for class_label in df["class"].unique():
            class_data = df[df["class"] == class_label]
            coordinates = (
                np.array(class_data["coordinates"].tolist())
                if not normalized
                else np.array(class_data["normalized_coordinates"].tolist())
            )
            plt.scatter(coordinates[:, 0], coordinates[:, 1], label=class_label, s=10)

        plt.title("Extracted dataset")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.show()


# OLD IMPLEMENTATION
# class SyntheticDataset:
#     def __init__(self, image_path, color_map, tolerance=10):
#         self.image_path = image_path
#         self.color_map = color_map
#         self.tolerance = tolerance

#     def get_dataset(self) -> pd.DataFrame:
#         data_values = self.dataset['coordinate']
#         return self.dataset

#     def get_image_path(self):
#         return self.image_path

#     def generate_data_from_img(self) -> pd.DataFrame:
#         # Open the image
#         img = Image.open(self.image_path)

#         # Convert the image to RGBA (if it's not already in that format)
#         img = img.convert("RGBA")

#         # Get the size of the image
#         width, height = img.size

#         # Prepare a list to hold coordinates and color of colored points
#         data = []

#         # Loop through each pixel in the image
#         for x in range(width):
#             for y in range(height):
#                 # Get the color of the pixel (excluding alpha channel for simplicity)
#                 r, g, b, _ = img.getpixel((x, y))
#                 if (r, g, b) == (237, 28, 36):
#                     data.append(((x, y), (r, g, b), "red"))
#                 if (r, g, b) == (255, 127, 29):
#                     data.append(((x, y), (r, g, b), "orange"))
#                 if (r, g, b) == (34, 177, 76):
#                     data.append(((x, y), (r, g, b), "green"))
#                 if (r, g, b) == (255, 242, 0) or (r, g, b) == (255, 201, 14):
#                     data.append(((x, y), (r, g, b), "yellow"))
#                 if (r, g, b) == (63, 72, 204):
#                     data.append(((x, y), (r, g, b), "blue"))

#         dataset = pd.DataFrame(data, columns=["coordinate", "rgb", "class"])
#         self.dataset = dataset

#     def one_hot_enc(self, column):
#         one_hot_df = pd.get_dummies(self.dataset, columns=[column])

#         return one_hot_df

#     def transform_to_numeric_classes(self, labels_col):
#         encoder = LabelEncoder()
#         df_numeric_classes = self.dataset.copy()
#         df_numeric_classes[f"{labels_col}"] = encoder.fit_transform(
#             df_numeric_classes[f"{labels_col}"]
#         )

#         return df_numeric_classes

#     def plot_data(self):
#         df = self.dataset

#         plt.figure(figsize=(8, 6))

#         for label in df["class"].unique():
#             class_data = df[df["class"] == label]
#             data_x = [coord[0] for coord in class_data["coordinate"]]
#             data_y = [coord[1] for coord in class_data["coordinate"]]

#             plt.scatter(
#                 data_x, data_y, label=f"Class {label}", alpha=0.8, s=15, marker="o"
#             )

#         # Optional: Invert y-axis to match the image's original coordinate system
#         # ax.invert_yaxis()

#         # Optional: Add legend and titles
#         plt.legend()
#         plt.title("Dataset")
#         plt.xlabel("X")
#         plt.ylabel("Y")
#         plt.show()

def make_clusters(n_samples_per_cluster=200, n_cluster=2, means=None, covs=None, seed=seed):
    means = (np.array([
        [0.2, 0.2],
        [0.9, 0.9]
    ]) if means is None else means)

    covs = ([
    [[0.1, 0],
     [0, 0.1]],
    
    [[0.1, 0],
     [0, 0.1]],
    ] if covs is None else covs)

    assert len(means) == len(covs) == n_cluster, "n_cluster must equal len(means) and len(covs)"

    n_samples = n_samples_per_cluster * n_cluster
    dim = np.shape(means)[1]
    data = []

    rng = np.random.default_rng(seed)
    for mean, cov in zip(means, covs):
        new_samples = rng.multivariate_normal(mean, cov, n_samples_per_cluster)
        data.extend(new_samples)
    
    return np.array(data)


def load_data(data_path: str, normalize=False) -> np.ndarray:
    if data_path.lower() == "random":
        # Generate random data
        np.random.seed(0)
        return np.random.rand(1000, 2)
    elif data_path.lower().endswith((".png", ".jpg", ".jpeg")):
        # Use SyntheticDataset for image-based data generation
        from utils.data_colors import COLOR_MAP

        extractor = SyntheticDataset(data_path, COLOR_MAP)
        dataset: pd.DataFrame = extractor.extract_data()

        if normalize:
            normalized_coordinates: np.ndarray = extractor.normalize_coordinates(
                dataset
            )
            return normalized_coordinates
    else:
        # Load data from file
        return np.loadtxt(data_path)


def create_save_path(base_path: str, algorithm: str) -> str:
    """
    Create a timestamped save path for the results.

    Args:
        base_path (str): Base directory for saving results.
        algorithm (str): Name of the algorithm being used.

    Returns:
        str: Full path to the save directory.
    """
    timestamp = datetime.now().strftime("%m-%d-%Y--%H-%M-%S")
    save_path = os.path.join(base_path, f"{algorithm.upper()}_{timestamp}")

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    return save_path
