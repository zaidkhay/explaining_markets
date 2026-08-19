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


class _PositiveTieCalibrator:
    """Mimic a monotonic empirical calibration that collapses upper neighbors."""

    def calibrate(self, value: float) -> float:
        if value < 0.5:
            return 0.30
        return 0.70


class _RawDirectionalButCalibratedTieModel:
    calibrator = _PositiveTieCalibrator()

    def predict_raw_vector(self, vector) -> float:
        return 0.5 + 0.05 * float(vector.values["eps_surprise_percent"])


def test_live_gate_passes_directional_realized_result_model():
    gate = evaluate_v3_lite_live_gate(
        _DirectionalModel(),
        min_submitted_spread=0.05,
        min_adjacent_submitted_gap=0.02,
    )
    assert gate.parsed_ok is True
    assert gate.zero_fls is True
    assert gate.ordered is True
    assert gate.submitted_spread > 0.05
    assert gate.negative_neutral_gap >= 0.02
    assert gate.neutral_positive_gap >= 0.02
    assert gate.passed is True


def test_live_gate_rejects_flat_model_even_when_parser_works():
    gate = evaluate_v3_lite_live_gate(_FlatModel(), min_submitted_spread=0.05)
    assert gate.parsed_ok is True
    assert gate.zero_fls is True
    assert gate.ordered is False
    assert gate.submitted_spread == 0.0
    assert gate.negative_neutral_gap == 0.0
    assert gate.neutral_positive_gap == 0.0
    assert gate.passed is False


def test_live_gate_rejects_calibration_tie_between_neutral_and_positive():
    gate = evaluate_v3_lite_live_gate(
        _RawDirectionalButCalibratedTieModel(),
        min_submitted_spread=0.05,
        min_adjacent_submitted_gap=0.02,
    )
    raw = [scenario.raw for scenario in gate.scenarios]
    submitted = [scenario.submitted for scenario in gate.scenarios]
    assert raw[0] < raw[1] < raw[2]
    assert submitted[0] < submitted[1] == submitted[2]
    assert gate.ordered is False
    assert gate.neutral_positive_gap == 0.0
    assert gate.passed is False
