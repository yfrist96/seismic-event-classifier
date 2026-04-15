"""
Plot station locations on a map of Israel with EQ/EX window counts and main faults.
Stations shown as red triangles at exact coordinates, with #ex|#eq label below.

Usage: python -m src.post_review.plot_station_map
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FAULTS_SHP = os.path.join(PROJECT_ROOT, "data", "mainfaultsisrael", "Main_Faults.shp")

EX_COLOR = '#3498db'  # blue
EQ_COLOR = '#e74c3c'  # red

# Israel TM Grid (ITM) — parameters from Main_Faults.prj
ITM_CRS = ccrs.TransverseMercator(
    central_longitude=35.20451694444445,
    central_latitude=31.73439361111111,
    false_easting=219529.584,
    false_northing=626907.39,
    scale_factor=1.0000067,
    globe=ccrs.Globe(ellipse='GRS80'),
)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "station_list.csv"))

    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_extent([34.0, 36.0, 29.3, 33.6], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='#f5f5f0', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#d4eaf7')
    ax.add_feature(cfeature.LAKES, facecolor='#a8d8ea', edgecolor='#6a9cb8', linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, edgecolor='#a8d8ea', linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#555555')

    # Main faults overlay (reproject from Israel TM to PlateCarree)
    faults = ShapelyFeature(Reader(FAULTS_SHP).geometries(),
                            ITM_CRS, edgecolor='#6b4226',
                            facecolor='none', linewidth=1.0)
    ax.add_feature(faults, zorder=4)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5,
                      color='gray', linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}

    # Station markers: red triangles at exact coords
    ax.scatter(df['longitude'], df['latitude'],
               marker='^', s=80, c='red', edgecolors='black',
               linewidths=0.7, zorder=6, transform=ccrs.PlateCarree())

    # Alternate label placement (above/below) per station; manual overrides below
    df_sorted = df.sort_values(['latitude', 'longitude']).reset_index(drop=True)
    bbox_kw = dict(boxstyle='round,pad=0.15', facecolor='white',
                   alpha=0.9, edgecolor='gray', linewidth=0.3)

    # Per-station overrides: placement = 'above' | 'below' | 'left' | 'right'
    PLACEMENT_OVERRIDES = {
        'BLGI': 'below',
        'SLTI': 'below',
        'KZIT': 'below',
        'MBRI': 'above',
        'EIL':  'below-left',
    }

    for i, row in df_sorted.iterrows():
        x, y = row['longitude'], row['latitude']
        station = row['station']
        override = PLACEMENT_OVERRIDES.get(station)

        if override in ('below', 'above', 'left', 'right', 'below-left'):
            placement = override
        else:
            placement = 'above' if (i % 2 == 0) else 'below'

        if placement == 'below-left':
            # Label's top-right corner touches triangle's bottom-left corner
            name_xy = (-5, -5)
            name_ha = 'right'
            name_va = 'top'
            count_dy = -16
            count_va = 'top'
            # Left-align each part starting from computed left anchor so the
            # whole "ex|eq" group ends at x=-5
            ex_w = len(str(row['explosions'])) * 4.2
            eq_w = len(str(row['earthquakes'])) * 4.2
            bar_w = 3.0
            gap = 2.0
            total_w = ex_w + gap + bar_w + gap + eq_w
            left_x = -5 - total_w
            ex_xy = (left_x, count_dy)
            bar_xy = (left_x + ex_w + gap, count_dy)
            eq_xy = (left_x + ex_w + gap + bar_w + gap, count_dy)
            ex_ha = bar_ha = eq_ha = 'left'
        elif placement in ('above', 'below'):
            above = (placement == 'above')
            name_dy = 14 if above else -14
            name_va = 'bottom' if above else 'top'
            count_dy = name_dy + (11 if above else -11)
            count_va = name_va
            name_xy = (0, name_dy)
            ex_xy = (-2, count_dy)
            bar_xy = (0, count_dy)
            eq_xy = (2, count_dy)
            name_ha = 'center'
            ex_ha, bar_ha, eq_ha = 'right', 'center', 'left'
        else:
            # left or right side placement
            sign = -1 if placement == 'left' else 1
            dx = sign * 12
            name_xy = (dx, 4)
            count_xy_base = (dx, -7)
            name_va = 'bottom'
            count_va = 'top'
            name_ha = 'right' if sign < 0 else 'left'
            ex_ha, bar_ha, eq_ha = name_ha, name_ha, name_ha
            # stack the three count texts horizontally next to each other
            # approximate widths: digits ~4pt each at fontsize 7
            if sign < 0:
                ex_xy = (dx - 10, -7)
                bar_xy = (dx - 4, -7)
                eq_xy = (dx, -7)
                ex_ha = bar_ha = eq_ha = 'right'
            else:
                ex_xy = (dx, -7)
                bar_xy = (dx + 6, -7)
                eq_xy = (dx + 10, -7)
                ex_ha = bar_ha = eq_ha = 'left'

        ax.annotate(station,
                    xy=(x, y), xytext=name_xy, textcoords="offset points",
                    ha=name_ha, va=name_va, fontsize=7, fontweight='bold',
                    bbox=bbox_kw, transform=ccrs.PlateCarree(), zorder=7)

        ax.annotate(f"{row['explosions']}",
                    xy=(x, y), xytext=ex_xy, textcoords="offset points",
                    ha=ex_ha, va=count_va, fontsize=7,
                    color=EX_COLOR, fontweight='bold',
                    transform=ccrs.PlateCarree(), zorder=8)
        ax.annotate("|",
                    xy=(x, y), xytext=bar_xy, textcoords="offset points",
                    ha=bar_ha, va=count_va, fontsize=7, color='black',
                    transform=ccrs.PlateCarree(), zorder=8)
        ax.annotate(f"{row['earthquakes']}",
                    xy=(x, y), xytext=eq_xy, textcoords="offset points",
                    ha=eq_ha, va=count_va, fontsize=7,
                    color=EQ_COLOR, fontweight='bold',
                    transform=ccrs.PlateCarree(), zorder=8)

    legend_handles = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
               markeredgecolor='black', markersize=9, label='Station'),
        Line2D([0], [0], color='#6b4226', linewidth=1.2, label='Main faults'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=EX_COLOR,
               markersize=9, label='#explosions'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=EQ_COLOR,
               markersize=9, label='#earthquakes'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=9,
              framealpha=0.9, edgecolor='gray')

    ax.set_title("Stations and Event Counts over Israel's Main Faults",
                 fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "station_map")
    fig.savefig(out_path + ".png", dpi=300, bbox_inches='tight')
    fig.savefig(out_path + ".pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}.png + .pdf")
