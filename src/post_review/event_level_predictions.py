"""
Aggregate window-level test predictions to event-level using majority vote.

Reads output/test_predictions.csv and produces output/event_level_predictions.csv
with one row per event, plus a printed summary comparing window-level vs event-level
accuracy across all prediction types (single, calibrated, ensemble).

Usage: python -m src.post_review.event_level_predictions
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import mode

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

LABEL_MAP = {0: 'earthquake', 1: 'explosion'}


def aggregate_event(group):
    """Aggregate all windows for a single event into one event-level prediction."""
    event_id = group['event_id'].iloc[0]
    true_label = group['true_label'].iloc[0]
    n_windows = len(group)
    stations = ','.join(sorted(group['station'].unique()))

    row = {
        'event_id': event_id,
        'true_label': true_label,
        'true_class': LABEL_MAP[true_label],
        'n_windows': n_windows,
        'n_stations': group['station'].nunique(),
        'stations': stations,
    }

    # Aggregate each prediction type by majority vote across windows
    for pred_type in ['single', 'calibrated', 'ensemble']:
        label_col = f'{pred_type}_pred_label'
        preds = group[label_col].values
        majority_label = int(mode(preds, keepdims=False).mode)
        vote_frac = np.mean(preds == majority_label)

        row[f'{pred_type}_event_pred'] = majority_label
        row[f'{pred_type}_event_class'] = LABEL_MAP[majority_label]
        row[f'{pred_type}_window_agreement'] = round(vote_frac, 3)

    # Mean ensemble decision score across all windows
    row['mean_decision_score'] = round(group['ensemble_mean_decision_score'].mean(), 4)

    return pd.Series(row)


def main():
    input_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} windows from {df['event_id'].nunique()} events\n")

    # Aggregate to event level
    event_df = df.groupby('event_id').apply(aggregate_event, include_groups=False).reset_index(drop=True)
    event_df['event_id'] = event_df['event_id'].astype(int)
    event_df['true_label'] = event_df['true_label'].astype(int)
    for pred_type in ['single', 'calibrated', 'ensemble']:
        event_df[f'{pred_type}_event_pred'] = event_df[f'{pred_type}_event_pred'].astype(int)

    # Save
    output_path = os.path.join(OUTPUT_DIR, "event_level_predictions.csv")
    event_df.to_csv(output_path, index=False)
    print(f"Saved {len(event_df)} event-level predictions to: {output_path}\n")

    # Print comparison: window-level vs event-level accuracy
    print("=" * 60)
    print(f"{'Prediction Type':<20} {'Window Acc':>12} {'Event Acc':>12} {'Delta':>8}")
    print("-" * 60)

    y_true_window = df['true_label'].values
    y_true_event = event_df['true_label'].values

    for pred_type in ['single', 'calibrated', 'ensemble']:
        window_acc = np.mean(df[f'{pred_type}_pred_label'].values == y_true_window)
        event_acc = np.mean(event_df[f'{pred_type}_event_pred'].values == y_true_event)
        delta = event_acc - window_acc
        print(f"{pred_type:<20} {window_acc:>11.2%} {event_acc:>11.2%} {delta:>+7.2%}")

    print("=" * 60)

    # Per-class breakdown for ensemble (event level)
    print(f"\nEnsemble event-level per-class breakdown:")
    print("-" * 40)
    for label, name in LABEL_MAP.items():
        mask = y_true_event == label
        n = mask.sum()
        correct = np.sum(event_df.loc[mask, 'ensemble_event_pred'].values == label)
        print(f"  {name:<12} {correct}/{n} correct ({correct/n:.2%})")

    # List misclassified events
    ens_wrong = event_df[event_df['ensemble_event_pred'] != event_df['true_label']]
    if len(ens_wrong) > 0:
        print(f"\nMisclassified events (ensemble, n={len(ens_wrong)}):")
        print("-" * 70)
        for _, row in ens_wrong.iterrows():
            print(f"  Event {int(row['event_id']):>6d} | "
                  f"true={row['true_class']:<11s} pred={row['ensemble_event_class']:<11s} | "
                  f"{int(row['n_windows'])} windows, "
                  f"agreement={row['ensemble_window_agreement']:.0%}, "
                  f"score={row['mean_decision_score']:+.3f}")


if __name__ == "__main__":
    main()
