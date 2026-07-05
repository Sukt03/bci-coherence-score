from bci_repro.agreement import cohen_kappa, fleiss_kappa


def test_cohen_kappa_perfect():
    labels = ["yes", "somewhat", "no", "yes"]
    assert cohen_kappa(labels, labels) == 1.0
    assert cohen_kappa(labels, labels, weighted=True) == 1.0


def test_fleiss_kappa_runs():
    items = [
        {"a": "yes", "b": "yes", "c": "somewhat"},
        {"a": "no", "b": "no", "c": "no"},
        {"a": "somewhat", "b": "yes", "c": "somewhat"},
    ]
    value = fleiss_kappa(items, ["a", "b", "c"])
    assert value is not None
    assert -1.0 <= value <= 1.0

