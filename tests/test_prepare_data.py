"""
test_prepare_data.py
Unit tests for data preprocessing pipeline, standardization, and dataset integrity.
"""

import os
import json
import numpy as np
import pytest
from src.prepare_data import (
    INPUT_COLUMNS,
    TARGET_COLUMNS,
    standardize,
    destandardize,
    process_and_save
)


def test_processed_files_exist():
    """
    Verify all expected artifacts are saved under outputs/processed/.
    """
    expected_files = [
        'outputs/processed/normalization_stats.json',
        'outputs/processed/train_data.npz',
        'outputs/processed/test_data.npz',
        'outputs/processed/preprocessing_metadata.md'
    ]
    for path in expected_files:
        assert os.path.isfile(path), f"Missing artifact: {path}"


def test_training_data_shapes_and_statistics():
    """
    Verify training dataset shapes, zero missing values, and proper standardization (mean ~ 0, std ~ 1).
    """
    data = np.load('outputs/processed/train_data.npz')

    inputs_raw = data['inputs_raw']
    targets_raw = data['targets_raw']
    inputs_norm = data['inputs_norm']
    targets_norm = data['targets_norm']

    assert inputs_raw.shape == (2997, 4)
    assert targets_raw.shape == (2997, 3)
    assert inputs_norm.shape == (2997, 4)
    assert targets_norm.shape == (2997, 3)

    assert np.all(np.isfinite(inputs_raw))
    assert np.all(np.isfinite(targets_raw))
    assert np.all(np.isfinite(inputs_norm))
    assert np.all(np.isfinite(targets_norm))

    # Normalized training data should have mean ~ 0 and sample std (ddof=1) == 1
    np.testing.assert_allclose(inputs_norm.mean(axis=0), np.zeros(4), atol=1e-5)
    np.testing.assert_allclose(inputs_norm.std(axis=0, ddof=1), np.ones(4), atol=1e-5)

    np.testing.assert_allclose(targets_norm.mean(axis=0), np.zeros(3), atol=1e-5)
    np.testing.assert_allclose(targets_norm.std(axis=0, ddof=1), np.ones(3), atol=1e-5)


def test_standardization_invertibility():
    """
    Verify that destandardize(standardize(x)) perfectly recovers x.
    """
    data = np.load('outputs/processed/train_data.npz')
    inputs_raw = data['inputs_raw']
    in_mean = data['in_mean']
    in_std = data['in_std']

    norm = standardize(inputs_raw, in_mean, in_std)
    recovered = destandardize(norm, in_mean, in_std)

    np.testing.assert_allclose(recovered, inputs_raw, rtol=1e-5, atol=1e-6)


def test_test_data_contents():
    """
    Verify that test_data.npz contains all 3 test maneuvers with 1000 steps each.
    """
    test_data = np.load('outputs/processed/test_data.npz')

    for maneuver in ['turning_circle_30', 'zigzag_20_10', 'zigzag_25_5']:
        assert f'{maneuver}_u' in test_data
        assert f'{maneuver}_v' in test_data
        assert f'{maneuver}_r' in test_data
        assert f'{maneuver}_rudder_angle' in test_data
        assert len(test_data[f'{maneuver}_u']) == 1000


def test_normalization_stats_json():
    """
    Verify normalization_stats.json has correct structure and non-zero stds.
    """
    with open('outputs/processed/normalization_stats.json', 'r') as f:
        stats = json.load(f)

    assert 'inputs' in stats
    assert 'targets' in stats

    for col in INPUT_COLUMNS:
        assert col in stats['inputs']
        assert stats['inputs'][col]['std'] > 0.0

    for col in TARGET_COLUMNS:
        assert col in stats['targets']
        assert stats['targets'][col]['std'] > 0.0
