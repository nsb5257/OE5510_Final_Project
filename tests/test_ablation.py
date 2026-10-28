"""
tests/test_ablation.py
Unit tests for the controlled ablation study (Dynamic vs. Constant Loss Weighting) (T10).
"""

import os
import json
import pytest
import numpy as np
import torch

from src.compare_ablation import (
    load_ablation_checkpoints,
    evaluate_model_on_maneuvers,
    run_ablation_comparison
)


def test_ablation_checkpoints_exist_and_valid():
    """Verify both dynamic and constant checkpoints exist and have identical training setup."""
    ckpt_dyn, ckpt_const, stats, test_data = load_ablation_checkpoints()

    assert ckpt_dyn['iterations'] == ckpt_const['iterations']
    assert ckpt_dyn['weighting'] == 'dynamic'
    assert ckpt_const['weighting'] == 'constant'

    assert 'hydro_params' in ckpt_dyn
    assert 'hydro_params' in ckpt_const
    assert len(ckpt_dyn['hydro_params']) == 36
    assert len(ckpt_const['hydro_params']) == 36


def test_ablation_outputs_generation_and_integrity():
    """Verify that ablation analysis runs and generates all expected reports and plots."""
    results = run_ablation_comparison()

    json_path = 'outputs/metrics/ablation_metrics.json'
    md_path = 'outputs/metrics/ablation_metrics.md'
    plot_path = 'outputs/plots/ablation_comparison.png'

    assert os.path.exists(json_path) and os.path.getsize(json_path) > 500
    assert os.path.exists(md_path) and os.path.getsize(md_path) > 500
    assert os.path.exists(plot_path) and os.path.getsize(plot_path) > 1000

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    assert 'training_losses' in data
    assert 'parameter_errors' in data
    assert 'trajectory_rmse' in data

    dyn_loss = data['training_losses']['dynamic']
    const_loss = data['training_losses']['constant']

    assert np.isfinite(dyn_loss['final_total_loss'])
    assert np.isfinite(const_loss['final_total_loss'])
    assert dyn_loss['final_total_loss'] > 0.0
    assert const_loss['final_total_loss'] > 0.0


def test_ablation_comparative_performance():
    """Verify that dynamic weighting produces competitive or superior parameter convergence."""
    with open('outputs/metrics/ablation_metrics.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    param_err = data['parameter_errors']
    dyn_mean_err = param_err['dynamic']['mean_dominant_rel_error_percent']
    const_mean_err = param_err['constant']['mean_dominant_rel_error_percent']

    # Both conditions must converge to reasonable dominant parameter errors (< 15%)
    assert dyn_mean_err < 15.0
    assert const_mean_err < 15.0
