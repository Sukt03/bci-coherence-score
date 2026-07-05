import pandas as pd

from bci_repro.metric_utils import corr_value, oriented_quality, quartile_failure_rates


def test_oriented_quality_flips_lower_is_better():
    values = pd.Series([1.0, 2.0, 3.0])
    assert oriented_quality(values, "mse").tolist() == [-1.0, -2.0, -3.0]
    assert oriented_quality(values, "ssim").tolist() == [1.0, 2.0, 3.0]


def test_corr_value_spearman():
    assert corr_value(pd.Series([1, 2, 3]), pd.Series([10, 20, 30])) == 1.0


def test_quartile_failure_rates_shape():
    rates = quartile_failure_rates(
        pd.Series([0.1, 0.2, 0.9, 1.0]),
        pd.Series([1.0, 1.0, 0.0, 0.0]),
        "ssim",
        semantic_threshold=0.5,
    )
    assert rates["n"] == 4
    assert rates["harshness_rate"] is not None
    assert rates["blind_spot_rate"] is not None

