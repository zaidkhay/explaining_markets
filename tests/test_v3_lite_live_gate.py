from explaining_markets.v3_lite_live_gate import evaluate_v3_lite_live_gate


class _IdentityCalibrator:
    n_fitted = 0

    def calibrate(self, value: float) -> float:
        return float(value)


class _RankAwareIdentityCalibrator:
    n_fitted = 1000

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
    n_fitted = 1000

    def calibrate(self, value: float) -> float:
        if value < 0.5:
            return 0.30
        return 0.70


class _RawDirectionalButCalibratedTieModel:
    calibrator = _PositiveTieCalibrator()

    def predict_raw_vector(self, vector) -> float:
        return 0.5 + 0.05 * float(vector.values["eps_surprise_percent"])


class _SmallButMultiRankDirectionalModel:
    calibrator = _RankAwareIdentityCalibrator()

    def predict_raw_vector(self, vector) -> float:
        # +/-12% EPS surprise becomes +/-0.006 score around neutral, i.e.
        # adjacent gaps of 0.006. With 1000 OOS calibration rows that is six
        # historical rank steps, so it should pass a five-rank requirement.
        return 0.5 + 0.05 * float(vector.values["eps_surprise_percent"])


def test_live_gate_passes_directional_realized_result_model():
    gate = evaluate_v3_lite_live_gate(
        _DirectionalModel(),
        min_submitted_spread=0.05,
        min_adjacent_rank_steps=5,
    )
    assert gate.parsed_ok is True
    assert gate.zero_fls is True
    assert gate.ordered is True
    assert gate.submitted_spread > 0.05
    assert gate.minimum_adjacent_gap_required == 0.005
    assert gate.negative_neutral_gap >= gate.minimum_adjacent_gap_required
    assert gate.neutral_positive_gap >= gate.minimum_adjacent_gap_required
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
        min_adjacent_rank_steps=5,
    )
    raw = [scenario.raw for scenario in gate.scenarios]
    submitted = [scenario.submitted for scenario in gate.scenarios]
    assert raw[0] < raw[1] < raw[2]
    assert submitted[0] < submitted[1] == submitted[2]
    assert gate.ordered is False
    assert gate.neutral_positive_gap == 0.0
    assert gate.passed is False


def test_live_gate_uses_oos_rank_resolution_not_arbitrary_two_percent_floor():
    gate = evaluate_v3_lite_live_gate(
        _SmallButMultiRankDirectionalModel(),
        min_submitted_spread=0.005,
        min_adjacent_rank_steps=5,
    )
    assert gate.calibration_n_fitted == 1000
    assert gate.minimum_adjacent_gap_required == 0.005
    assert gate.negative_neutral_gap > 0.005
    assert gate.neutral_positive_gap > 0.005
    assert gate.passed is True
