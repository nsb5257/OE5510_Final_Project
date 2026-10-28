"""
tests/test_analyze_derivatives.py
Unit tests for hydrodynamic derivative analysis and comparison (T9).
"""

import os
import json
import pytest
import numpy as np

from src.analyze_derivatives import (
    load_identified_checkpoint,
    analyze_hydrodynamic_derivatives,
    PAPER_IDENTIFIED_DERIVATIVES,
    DOMINANT_PARAMETERS,
    PARAMETER_GROUPS
)
from src.ship_dynamics import ORIGINAL_HYDRODYNAMIC_DERIVATIVES


def test_parameter_counts_and_definitions():
    """Verify that all 36 derivatives are accounted for across all sources."""
    assert len(ORIGINAL_HYDRODYNAMIC_DERIVATIVES) == 36
    assert len(PAPER_IDENTIFIED_DERIVATIVES) == 36

    total_grouped = sum(len(params) for params in PARAMETER_GROUPS.values())
    assert total_grouped == 36

    for p in ORIGINAL_HYDRODYNAMIC_DERIVATIVES:
        assert p in PAPER_IDENTIFIED_DERIVATIVES


def test_load_identified_checkpoint():
    """Verify loading identified parameters from pinn_dynamic.pt."""
    params, iters, train_time = load_identified_checkpoint('outputs/checkpoints/pinn_dynamic.pt')
    assert len(params) == 36
    assert iters == 20000
    assert train_time > 0.0

    for p in ORIGINAL_HYDRODYNAMIC_DERIVATIVES:
        assert p in params
        assert np.isfinite(params[p])


def test_analyze_derivatives_execution_and_outputs():
    """Verify analysis execution and output file generation."""
    results = analyze_hydrodynamic_derivatives()
    assert len(results) == 36

    md_path = 'outputs/metrics/hydrodynamic_derivatives_comparison.md'
    json_path = 'outputs/metrics/hydrodynamic_derivatives.json'
    plot_path = 'outputs/plots/derivatives_comparison.png'

    assert os.path.exists(md_path) and os.path.getsize(md_path) > 500
    assert os.path.exists(json_path) and os.path.getsize(json_path) > 500
    assert os.path.exists(plot_path) and os.path.getsize(plot_path) > 1000

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    assert 'metadata' in data
    assert 'derivatives' in data
    assert len(data['derivatives']) == 36


def test_dominant_parameters_accuracy():
    """Verify that dominant linear/quadratic derivatives have high accuracy (< 2% error)."""
    with open('outputs/metrics/hydrodynamic_derivatives.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    derivs = data['derivatives']

    # Primary linear damping & control derivatives
    assert derivs['Xu_prime']['rel_error_true_percent'] < 2.0
    assert derivs['Xuu_prime']['rel_error_true_percent'] < 2.0
    assert derivs['Yv_prime']['rel_error_true_percent'] < 2.0
    assert derivs['Nr_prime']['rel_error_true_percent'] < 2.0
    assert derivs['Yd_prime']['rel_error_true_percent'] < 2.0
    assert derivs['Nd_prime']['rel_error_true_percent'] < 2.0

    # Overall dominant group mean error < 3%
    dom_errors = [derivs[k]['rel_error_true_percent'] for k in DOMINANT_PARAMETERS]
    assert np.mean(dom_errors) < 3.0
