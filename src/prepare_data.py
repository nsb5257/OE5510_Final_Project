"""
prepare_data.py
Minimal preprocessing pipeline for ship maneuvering reproduction.
Reference:
  "Identification modeling of ship maneuvering motion based on physics-informed neural network"
  (Hu et al., 2026, Section 4.1, Equation 19)
"""

import os
import json
import numpy as np
import pandas as pd

TRAIN_FILES = [
    'data/train_data/turning_circle_35.txt',
    'data/train_data/zigzag_10_10.txt',
    'data/train_data/zigzag_20_20.txt'
]

TEST_FILES = [
    'data/test_data/turning_circle_30.txt',
    'data/test_data/zigzag_20_10.txt',
    'data/test_data/zigzag_25_5.txt'
]

INPUT_COLUMNS = ['u', 'v', 'r', 'rudder_angle']
TARGET_COLUMNS = ['du_dt', 'dv_dt', 'dr_dt']


def load_raw_training_data():
    """
    Loads all training files and concatenates them into a single dataframe,
    recording maneuver boundaries to preserve trajectory context.
    """
    dfs = []
    maneuver_info = {}
    current_idx = 0

    for path in TRAIN_FILES:
        name = os.path.basename(path).replace('.txt', '')
        df = pd.read_csv(path, sep=r'\s+')
        df['maneuver'] = name
        n_rows = len(df)
        maneuver_info[name] = (current_idx, current_idx + n_rows)
        current_idx += n_rows
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df, maneuver_info


def compute_normalization_stats(train_df):
    """
    Computes mean (mu) and standard deviation (sigma) for input and target
    variables on training data strictly (preventing data leakage).
    Reference: Page 6, Equation (19).
    """
    stats = {
        'inputs': {},
        'targets': {}
    }

    for col in INPUT_COLUMNS:
        stats['inputs'][col] = {
            'mean': float(train_df[col].mean()),
            'std': float(train_df[col].std())
        }

    for col in TARGET_COLUMNS:
        stats['targets'][col] = {
            'mean': float(train_df[col].mean()),
            'std': float(train_df[col].std())
        }

    return stats


def standardize(data, mean, std):
    """
    Applies z-score standardization: x_norm = (x - mu) / sigma
    Reference: Page 6, Equation (19).
    """
    std_safe = np.where(std == 0.0, 1.0, std)
    return (data - mean) / std_safe


def destandardize(norm_data, mean, std):
    """
    Inverts z-score standardization: x = norm_data * sigma + mu
    """
    return norm_data * std + mean


def load_raw_test_data():
    """
    Loads the 3 held-out test maneuvers.
    Preserves all trajectory coordinates and controls.
    """
    test_dict = {}
    for path in TEST_FILES:
        name = os.path.basename(path).replace('.txt', '')
        df = pd.read_csv(path, sep=r'\s+')
        test_dict[name] = df
    return test_dict


def process_and_save():
    os.makedirs('outputs/processed', exist_ok=True)

    # 1. Load training data and compute stats (no data leakage)
    train_df, maneuver_info = load_raw_training_data()
    stats = compute_normalization_stats(train_df)

    # Extract arrays
    inputs_raw = train_df[INPUT_COLUMNS].to_numpy(dtype=np.float32)
    targets_raw = train_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)

    in_mean = np.array([stats['inputs'][c]['mean'] for c in INPUT_COLUMNS], dtype=np.float32)
    in_std = np.array([stats['inputs'][c]['std'] for c in INPUT_COLUMNS], dtype=np.float32)
    tgt_mean = np.array([stats['targets'][c]['mean'] for c in TARGET_COLUMNS], dtype=np.float32)
    tgt_std = np.array([stats['targets'][c]['std'] for c in TARGET_COLUMNS], dtype=np.float32)

    inputs_norm = standardize(inputs_raw, in_mean, in_std)
    targets_norm = standardize(targets_raw, tgt_mean, tgt_std)

    # Save stats JSON
    stats_json_path = 'outputs/processed/normalization_stats.json'
    with open(stats_json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f'Saved stats: {stats_json_path}')

    # Save training dataset
    train_npz_path = 'outputs/processed/train_data.npz'
    np.savez_compressed(
        train_npz_path,
        inputs_raw=inputs_raw,
        targets_raw=targets_raw,
        inputs_norm=inputs_norm,
        targets_norm=targets_norm,
        in_mean=in_mean,
        in_std=in_std,
        tgt_mean=tgt_mean,
        tgt_std=tgt_std,
        input_columns=np.array(INPUT_COLUMNS),
        target_columns=np.array(TARGET_COLUMNS),
        maneuver_names=np.array(list(maneuver_info.keys())),
        maneuver_starts=np.array([v[0] for v in maneuver_info.values()]),
        maneuver_ends=np.array([v[1] for v in maneuver_info.values()])
    )
    print(f'Saved training dataset: {train_npz_path} (shape: {inputs_raw.shape})')

    # Save test dataset
    test_dict = load_raw_test_data()
    test_save_kwargs = {}
    for name, df in test_dict.items():
        for col in df.columns:
            key = f'{name}_{col.replace("-", "_")}'
            test_save_kwargs[key] = df[col].to_numpy(dtype=np.float32)

    test_npz_path = 'outputs/processed/test_data.npz'
    np.savez_compressed(test_npz_path, **test_save_kwargs)
    print(f'Saved test dataset: {test_npz_path}')

    # Write documentation metadata markdown
    meta_path = 'outputs/processed/preprocessing_metadata.md'
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write('# Preprocessing Metadata & Normalization Specification\n\n')
        f.write('**Generated:** 27 October 2026\n')
        f.write('**Source:** `src/prepare_data.py`\n')
        f.write('**Paper Reference:** Page 6, Section 4.1, Equation (19)\n\n')

        f.write('## 1. Train / Test Split Strategy\n\n')
        f.write('Trajectory-level partition (no random row mixing, preventing time-series leakage):\n')
        f.write('- **Training Maneuvers (2,997 steps total):**\n')
        for name, (s, e) in maneuver_info.items():
            f.write(f'  - `{name}`: indices [{s}, {e}) ({e - s} rows)\n')
        f.write('- **Held-out Test Maneuvers (1,000 steps each):**\n')
        for name in test_dict.keys():
            f.write(f'  - `{name}`: 1,000 seconds\n')

        f.write('\n## 2. Feature & Target Ordering\n\n')
        f.write('- **Input Vector ($X_{in}$):** `[u, v, r, rudder_angle]` (4 features)\n')
        f.write('- **Target Acceleration Vector ($Y_{out}$):** `[du_dt, dv_dt, dr_dt]` (3 targets)\n\n')

        f.write('## 3. Normalization Statistics (Computed on Training Data Only)\n\n')
        f.write('Standardization formula: $x_{norm} = \\frac{x - \\mu}{\\sigma}$\n\n')
        f.write('| Variable | Role | Physical Unit | Mean ($\\mu$) | Std Dev ($\\sigma$) |\n')
        f.write('|---|---|---|---|---|\n')
        for col in INPUT_COLUMNS:
            m = stats['inputs'][col]['mean']
            s = stats['inputs'][col]['std']
            f.write(f'| `{col}` | Input | physical | {m:.6e} | {s:.6e} |\n')
        for col in TARGET_COLUMNS:
            m = stats['targets'][col]['mean']
            s = stats['targets'][col]['std']
            f.write(f'| `{col}` | Target | physical | {m:.6e} | {s:.6e} |\n')

        f.write('\n## 4. Treatment of Physical Units in Loss Calculation\n\n')
        f.write('1. The neural network receives standardized inputs and outputs standardized accelerations.\n')
        f.write('2. Observation loss ($Loss_{data}$) compares predicted standardized accelerations with normalized target accelerations.\n')
        f.write('3. For physics-based loss ($Loss_{phy}$), predicted accelerations are destandardized: $\\dot{u} = \\dot{\\bar{u}} \\sigma + \\mu$.\n')
        f.write('4. Physical residuals $e_1, e_2, e_3$ are evaluated in physical units and normalized by target variance so that $Loss_{data}$ and $Loss_{phy}$ maintain balanced unitless scales.\n')

    print(f'Saved metadata: {meta_path}')


if __name__ == '__main__':
    process_and_save()
