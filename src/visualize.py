import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

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

def visualize_tsne_results():
    embedding_dir = os.path.join(CONFIG['output_dir'], "embeddings")
    plot_dir = os.path.join(CONFIG['output_dir'], "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n[t-SNE] Loading training labels from: {CONFIG['data_dir']}")
    # We need y_train to color the points correctly
    (X_train, y_train), _, _ = load_data(CONFIG['data_dir'])

    # Find ALL training embeddings (k=5, k=15, k=30, etc.)
    files = glob.glob(os.path.join(embedding_dir, "*_train.npy"))

    if not files:
        print("[t-SNE] No embeddings found! Run experiments first.")
        return

    print(f"[t-SNE] Found {len(files)} embedding files. generating plots...")

    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"  > Running t-SNE on {fname}...")

        # 1. Load the high-dimensional embedding (N, k)
        # e.g., Shape (N, 30)
        data = np.load(fpath)

        # 2. Run t-SNE to squash it down to 2D
        # We use a random_state for reproducibility
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        data_2d = tsne.fit_transform(data)

        # 3. Plot
        plt.figure(figsize=(10, 8))

        # Plot Earthquakes (Class 0) in Blue
        plt.scatter(data_2d[y_train == 0, 0], data_2d[y_train == 0, 1],
                    c='blue', label='Earthquake', alpha=0.5, s=15)
        
        # Plot Explosions (Class 1) in Red
        plt.scatter(data_2d[y_train == 1, 0], data_2d[y_train == 1, 1],
                    c='red', label='Explosion', alpha=0.5, s=15)

        title_str = fname.replace("_train.npy", "").replace("_", " ")
        plt.title(f"t-SNE Projection: {title_str}\n(Visualizing Clusters in {data.shape[1]} Dimensions)")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. Save with a distinct prefix
        save_name = "tsne_" + fname.replace(".npy", ".png")
        save_path = os.path.join(plot_dir, save_name)
        plt.savefig(save_path)
        plt.close()
        print(f"    Saved: {save_path}")

    print("[t-SNE] Visualization complete.\n")


if __name__ == "__main__":
    visualize_results()
    visualize_tsne_results()
    
    