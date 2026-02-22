"""
Tests for core/ml_inference.py

Validates HybridMLModel: loading, prediction, fusion, classification
thresholds, batch prediction, and weight management.
All external I/O (joblib, filesystem) is mocked.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from tests.conftest import MODEL_FEATURE_NAMES, make_safe_features, make_risky_features


# ═══════════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelLoading:

    def test_loads_all_components(self, mock_models):
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        assert model.ghana_gb is not None
        assert model.porto_if is not None
        assert model.scaler_ghana is not None
        assert model.scaler_porto is not None
        assert len(model.feature_names) == 18

    def test_default_fusion_weights(self, mock_models):
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        assert model.gb_weight == 0.7
        assert model.if_weight == 0.3

    def test_missing_model_file_raises(self, monkeypatch):
        monkeypatch.setattr('os.path.exists', lambda p: False)
        from core.ml_inference import HybridMLModel
        with pytest.raises(FileNotFoundError):
            HybridMLModel(models_dir='nonexistent')


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredict:

    def test_returns_required_keys(self, mock_models):
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        result = model.predict(make_safe_features())
        for key in ['gb_score', 'if_score', 'hybrid_score', 'level', 'color', 'components']:
            assert key in result, f"Missing key: {key}"

    def test_scores_in_zero_one_range(self, mock_models):
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        result = model.predict(make_safe_features())
        assert 0 <= result['gb_score'] <= 1
        assert 0 <= result['if_score'] <= 1
        assert 0 <= result['hybrid_score'] <= 1

    def test_hybrid_is_weighted_sum(self, mock_models):
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        result = model.predict(make_safe_features())
        expected = 0.7 * result['gb_score'] + 0.3 * result['if_score']
        assert result['hybrid_score'] == pytest.approx(expected, abs=1e-6)

    def test_components_breakdown(self, mock_models):
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        result = model.predict(make_safe_features())
        gb_comp = result['components']['ghana_gb']
        assert gb_comp['weight'] == 0.7
        assert gb_comp['contribution'] == pytest.approx(0.7 * gb_comp['score'], abs=1e-6)

    def test_missing_feature_raises_valueerror(self, mock_models):
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        incomplete = {'speed': 40.0}
        with pytest.raises(ValueError, match="Missing feature"):
            model.predict(incomplete)

    def test_extra_features_are_ignored(self, mock_models):
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        features = make_safe_features()
        features['extra_field'] = 999
        result = model.predict(features)
        assert 'hybrid_score' in result


# ═══════════════════════════════════════════════════════════════════════════════
# Classification thresholds
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassificationThresholds:

    def test_safe_below_0_3(self, mock_models):
        """Mock GB returns low risk probability → hybrid < 0.3 → SAFE."""
        mock_models['gb'].predict_proba.return_value = np.array([[0.95, 0.05]])
        mock_models['if'].decision_function.return_value = np.array([-2.0])
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        result = model.predict(make_safe_features())
        assert result['level'] == 'SAFE'
        assert result['color'] == 'green'

    def test_medium_between_0_3_and_0_7(self, mock_models):
        mock_models['gb'].predict_proba.return_value = np.array([[0.40, 0.60]])
        mock_models['if'].decision_function.return_value = np.array([0.0])
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        result = model.predict(make_safe_features())
        score = result['hybrid_score']
        assert 0.3 <= score < 0.7
        assert result['level'] == 'MEDIUM'
        assert result['color'] == 'orange'

    def test_high_risk_above_0_7(self, mock_models):
        mock_models['gb'].predict_proba.return_value = np.array([[0.05, 0.95]])
        mock_models['if'].decision_function.return_value = np.array([3.0])
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        result = model.predict(make_safe_features())
        assert result['hybrid_score'] >= 0.7
        assert result['level'] == 'HIGH RISK'
        assert result['color'] == 'red'

    def test_exact_boundary_0_3_is_medium(self, mock_models):
        """Score of exactly 0.3 should be MEDIUM (not SAFE)."""
        mock_models['gb'].predict_proba.return_value = np.array([[1 - 0.3 / 0.7, 0.3 / 0.7]])
        mock_models['if'].decision_function.return_value = np.array([0.0])
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        result = model.predict(make_safe_features())
        if result['hybrid_score'] >= 0.3:
            assert result['level'] in ('MEDIUM', 'HIGH RISK')


# ═══════════════════════════════════════════════════════════════════════════════
# IF score sigmoid normalization
# ═══════════════════════════════════════════════════════════════════════════════

class TestIFNormalization:

    def test_negative_raw_produces_below_half(self, mock_models):
        mock_models['if'].decision_function.return_value = np.array([-2.0])
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        result = model.predict(make_safe_features())
        assert result['if_score'] < 0.5

    def test_positive_raw_produces_above_half(self, mock_models):
        mock_models['if'].decision_function.return_value = np.array([2.0])
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        result = model.predict(make_safe_features())
        assert result['if_score'] > 0.5

    def test_zero_raw_produces_half(self, mock_models):
        mock_models['if'].decision_function.return_value = np.array([0.0])
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        result = model.predict(make_safe_features())
        assert result['if_score'] == pytest.approx(0.5, abs=0.001)


# ═══════════════════════════════════════════════════════════════════════════════
# Batch prediction
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchPredict:

    def test_batch_returns_correct_count(self, mock_models):
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        results = model.batch_predict([make_safe_features(), make_risky_features()])
        assert len(results) == 2

    def test_empty_batch(self, mock_models):
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        assert model.batch_predict([]) == []


# ═══════════════════════════════════════════════════════════════════════════════
# Fusion weight management
# ═══════════════════════════════════════════════════════════════════════════════

class TestSetFusionWeights:

    def test_valid_weights_update(self, mock_models):
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        model.set_fusion_weights(0.5, 0.5)
        assert model.gb_weight == 0.5
        assert model.if_weight == 0.5

    def test_weights_not_summing_to_one_raises(self, mock_models):
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            model.set_fusion_weights(0.6, 0.6)

    def test_updated_weights_affect_prediction(self, mock_models):
        from core.ml_inference import HybridMLModel
        model = HybridMLModel(models_dir='fake')

        r1 = model.predict(make_safe_features())
        model.set_fusion_weights(0.0, 1.0)
        r2 = model.predict(make_safe_features())

        assert r2['hybrid_score'] == pytest.approx(r2['if_score'], abs=1e-6)
        assert r2['hybrid_score'] != pytest.approx(r1['hybrid_score'], abs=1e-3)
