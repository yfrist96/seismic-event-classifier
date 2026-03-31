"""
Plot station locations on a map of Israel with EQ/EX window counts.
Each station shown as two offset bubbles (blue=EQ, red=EX) sized by count.
Uses cartopy for real coastlines and borders.

Usage: python -m src.post_review.plot_station_map
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Offset in degrees for the two bubbles (left=EQ, right=EX)
OFFSET = 0.08

# Bubble size scaling
MIN_SIZE = 40
MAX_SIZE = 350


def scale_sizes(counts, min_s=MIN_SIZE, max_s=MAX_SIZE):
    """Scale counts to marker sizes (area proportional to count)."""
    mn, mx = counts.min(), counts.max()
    if mx == mn:
        return np.full_like(counts, (min_s + max_s) / 2, dtype=float)
    return min_s + (counts - mn) / (mx - mn) * (max_s - min_s)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "station_list.csv"))

    fig = plt.figure(figsize=(7, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Map extent: Israel region
    ax.set_extent([34.0, 36.0, 29.3, 33.6], crs=ccrs.PlateCarree())

    # Cartopy features
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f0', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#d4eaf7')
    ax.add_feature(cfeature.LAKES, facecolor='#a8d8ea', edgecolor='#6a9cb8', linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, edgecolor='#a8d8ea', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.8, edgecolor='#888888')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#555555')

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5,
                      color='gray', linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}

    # Scale bubble sizes
    eq_sizes = scale_sizes(df['earthquakes'].values)
    ex_sizes = scale_sizes(df['explosions'].values)

    # Plot EQ bubbles (offset left)
    ax.scatter(df['longitude'] - OFFSET, df['latitude'],
               s=eq_sizes, c='#3498db', edgecolors='black', linewidths=0.6,
               alpha=0.85, zorder=5, transform=ccrs.PlateCarree(),
               label='Earthquake')

    # Plot EX bubbles (offset right)
    ax.scatter(df['longitude'] + OFFSET, df['latitude'],
               s=ex_sizes, c='#e74c3c', edgecolors='black', linewidths=0.6,
               alpha=0.85, zorder=5, transform=ccrs.PlateCarree(),
               label='Explosion')

    # Station labels with counts
    for _, row in df.iterrows():
        label = f"{row['station']}\n{row['earthquakes']}|{row['explosions']}"
        ax.annotate(label,
                    xy=(row['longitude'], row['latitude']),
                    xytext=(14, 10), textcoords="offset points",
                    fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.85, edgecolor='gray', linewidth=0.4),
                    transform=ccrs.PlateCarree(), zorder=6)

    # Legend with size reference
    legend = ax.legend(loc='upper left', fontsize=10, title='Event Type',
                       title_fontsize=10, framealpha=0.9, edgecolor='gray')

    ax.set_title('Station Locations and Event Distribution',
                 fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "station_map")
    fig.savefig(out_path + ".png", dpi=300, bbox_inches='tight')
    fig.savefig(out_path + ".pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}.png + .pdf")
