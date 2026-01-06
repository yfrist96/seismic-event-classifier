import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def visualize_results(output_dir="./output"):
    embedding_dir = os.path.join(output_dir, "embeddings")
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # We need labels. We can load them from the dataset again or save them in main.py.
    # Assuming main.py saved them, or we re-load:
    # (Simplified: Just re-load specific file needed)
    from src.dataloader import load_data
    (_, y_train), _, _ = load_data("./seismic_datasets")

    # Find all k=2 embeddings
    files = glob.glob(f"{embedding_dir}/*_k2_train.npy")

    for fpath in files:
        fname = os.path.basename(fpath)
        dist_name = "euclidean" if "euclidean" in fname else "correlation"

        print(f"Plotting {fname}...")
        data = np.load(fpath)

        plt.figure(figsize=(8, 6))

        # Class 0 = Earthquake, Class 1 = Explosion (Usually)
        plt.scatter(data[y_train == 0, 0], data[y_train == 0, 1],
                    c='blue', label='Earthquake', alpha=0.5, s=15)
        plt.scatter(data[y_train == 1, 0], data[y_train == 1, 1],
                    c='red', label='Explosion', alpha=0.5, s=15)

        plt.title(f"FastMap 2D Embedding ({dist_name})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(plot_dir, f"plot_{dist_name}_k2.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    visualize_results()