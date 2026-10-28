"""
tests/test_evaluate_all.py
Unit tests for full multi-maneuver evaluation and comparison pipeline (T8).
"""

import os
import json
import pytest
import numpy as np
import torch

from src.evaluate_all import (
    load_models_and_data,
    PAPER_REPORTED_METRICS,
    run_full_evaluation
)
from src.model import ShipMotionNN, ShipPINN


def test_paper_reported_metrics_structure():
    """Verify paper reported metrics dictionary structure matches Table 3."""
    expected_maneuvers = ['zigzag_25_5', 'zigzag_20_10', 'turning_circle_30']
    for m in expected_maneuvers:
        assert m in PAPER_REPORTED_METRICS
        assert 'PINN' in PAPER_REPORTED_METRICS[m]
        assert 'Identified' in PAPER_REPORTED_METRICS[m]
        for mod in ['PINN', 'Identified']:
            for key in ['u', 'v', 'r_deg', 'sum']:
                assert key in PAPER_REPORTED_METRICS[m][mod]
                assert PAPER_REPORTED_METRICS[m][mod][key] > 0.0


def test_load_models_and_data():
    """Verify loading baseline NN, PINN, identified params, normalization stats and test data."""
    baseline_model, pinn_model, ident_params, stats, test_data = load_models_and_data()

    assert isinstance(baseline_model, ShipMotionNN)
    assert isinstance(pinn_model, ShipPINN)
    assert isinstance(ident_params, dict)
    assert len(ident_params) == 36
    assert 'Xu_prime' in ident_params
    assert 'Yv_prime' in ident_params
    assert 'Nr_prime' in ident_params

    assert 'inputs' in stats
    assert 'targets' in stats

    for m in ['zigzag_25_5', 'zigzag_20_10', 'turning_circle_30']:
        assert f'{m}_u' in test_data
        assert f'{m}_v' in test_data
        assert f'{m}_r' in test_data
        assert f'{m}_rudder_angle' in test_data


def test_evaluation_outputs_exist():
    """Verify that evaluate_all produces valid figures, JSON metrics, and markdown table."""
    # Ensure full evaluation has been run
    metrics_json_path = 'outputs/metrics/model_comparison_metrics.json'
    table_md_path = 'outputs/metrics/model_comparison_table.md'
    fig6_path = 'outputs/plots/paper_fig6_zigzag_25_5_comparison.png'
    fig7_path = 'outputs/plots/paper_fig7_zigzag_20_10_comparison.png'
    fig8_path = 'outputs/plots/paper_fig8_turning_circle_30_comparison.png'

    assert os.path.exists(metrics_json_path), f"Missing {metrics_json_path}"
    assert os.path.exists(table_md_path), f"Missing {table_md_path}"
    assert os.path.exists(fig6_path), f"Missing {fig6_path}"
    assert os.path.exists(fig7_path), f"Missing {fig7_path}"
    assert os.path.exists(fig8_path), f"Missing {fig8_path}"

    with open(metrics_json_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    for m in ['zigzag_25_5', 'zigzag_20_10', 'turning_circle_30']:
        assert m in metrics
        for mod in ['PINN', 'Identified', 'Baseline']:
            assert mod in metrics[m]
            res = metrics[m][mod]
            assert res['rmse_u_m_s'] > 0.0 and np.isfinite(res['rmse_u_m_s'])
            assert res['rmse_v_m_s'] > 0.0 and np.isfinite(res['rmse_v_m_s'])
            assert res['rmse_r_deg_s'] > 0.0 and np.isfinite(res['rmse_r_deg_s'])
            assert res['sum_deg'] > 0.0 and np.isfinite(res['sum_deg'])


def test_identified_model_accuracy():
    """Verify that the identified hydrodynamic parameters achieve low simulation error."""
    with open('outputs/metrics/model_comparison_metrics.json', 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    # The identified model should achieve turning circle error < 0.05 m/s
    assert metrics['turning_circle_30']['Identified']['rmse_u_m_s'] < 0.05
    assert metrics['turning_circle_30']['Identified']['rmse_v_m_s'] < 0.02
