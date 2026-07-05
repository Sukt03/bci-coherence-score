from bci_repro.splits import SplitItem, split_indices


def test_split_indices_is_deterministic():
    examples = [
        SplitItem(pair_id=f"p{i}", concept=f"c{i // 2}", subject="s", method="m")
        for i in range(20)
    ]
    left = split_indices(examples, "concept", seed=42)
    right = split_indices(examples, "concept", seed=42)
    assert left == right
    assert sorted(left["train"] + left["val"] + left["test"]) == list(range(20))

