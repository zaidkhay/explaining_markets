from explaining_markets.v3_lite_live_gate import evaluate_v3_lite_live_gate


class _IdentityCalibrator:
    def calibrate(self, value: float) -> float:
        return float(value)


class _DirectionalModel:
    calibrator = _IdentityCalibrator()

    def predict_raw_vector(self, vector) -> float:
        return 0.5 + float(vector.values["eps_surprise_percent"])


class _FlatModel:
    calibrator = _IdentityCalibrator()

    def predict_raw_vector(self, vector) -> float:
        return 0.5


def test_live_gate_passes_directional_realized_result_model():
    gate = evaluate_v3_lite_live_gate(_DirectionalModel(), min_submitted_spread=0.05)
    assert gate.parsed_ok is True
    assert gate.zero_fls is True
    assert gate.ordered is True
    assert gate.submitted_spread > 0.05
    assert gate.passed is True


def test_live_gate_rejects_flat_model_even_when_parser_works():
    gate = evaluate_v3_lite_live_gate(_FlatModel(), min_submitted_spread=0.05)
    assert gate.parsed_ok is True
    assert gate.zero_fls is True
    assert gate.ordered is False
    assert gate.submitted_spread == 0.0
    assert gate.passed is False
