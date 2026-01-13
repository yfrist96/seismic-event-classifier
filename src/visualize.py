import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataloader import load_data
from src.config import CONFIG


def visualize_results():
    embedding_dir = os.path.join(CONFIG['output_dir'], "embeddings")
    plot_dir = os.path.join(CONFIG['output_dir'], "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Loading labels from: {CONFIG['data_dir']}")
    # Load ONLY training labels to match the training embeddings
    (X_train, y_train), _, _ = load_data(CONFIG['data_dir'])

    # Find ALL training embeddings (k=30, k=50, etc.)
    files = glob.glob(os.path.join(embedding_dir, "*_train.npy"))

    if not files:
        print("No embeddings found! Run main.py first.")
        return

    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"Plotting {fname}...")

        # Load embedding (N, k)
        data = np.load(fpath)

        # We only plot the first 2 dimensions (Dim 0 vs Dim 1)
        # regardless of whether k=2 or k=60
        plt.figure(figsize=(8, 6))

        # Class 0 = Earthquake, Class 1 = Explosion
        plt.scatter(data[y_train == 0, 0], data[y_train == 0, 1],
                    c='blue', label='Earthquake', alpha=0.5, s=15)
        plt.scatter(data[y_train == 1, 0], data[y_train == 1, 1],
                    c='red', label='Explosion', alpha=0.5, s=15)

        title_str = fname.replace("_train.npy", "").replace("_", " ")
        plt.title(f"First 2 Dimensions: {title_str}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_name = fname.replace(".npy", ".png")
        save_path = os.path.join(plot_dir, save_name)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    visualize_results()