"""D-033: the real CC-BY training data uses exactly the D-025 label maps of the test candidates."""

from danalm.config import load_config


def test_real_data_uses_the_test_candidate_label_maps_on_the_train_splits():
    real = {s["name"]: s for s in load_config("configs/sft/real.yaml")["fetch"]["sources"]}
    test = {
        s["name"]: s
        for s in load_config("configs/data/test_candidates_en.yaml")["fetch"]["sources"]
    }
    for name in ("banking77", "clinc150"):
        r, t = real[f"{name}-train"], test[f"{name}-test"]
        assert r["label_map"] == t["label_map"]
        assert r["revision"] == t["revision"]
        assert "test" not in (r.get("url") or r.get("files"))
