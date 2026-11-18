import json


def test_models_beat_baseline_and_blend_helps(trained):
    rep = json.loads((trained / "training_report.json").read_text())
    t = rep["gnn_blend"]["test"]
    base = t["xgb"]["base_rate"]
    assert t["xgb"]["pr_auc"] > 5 * base, "XGBoost must be far better than random (PR-AUC = base rate)"
    assert t["blend"]["pr_auc"] >= t["xgb"]["pr_auc"] - 0.01, "blending must not hurt"
    assert t["gnn"]["pr_auc"] > 2 * base, "GNN ring score must carry signal on its own"


def test_threshold_chosen_on_validation_generalises(trained):
    rep = json.loads((trained / "training_report.json").read_text())
    at = rep["gnn_blend"]["test_at_threshold"]
    assert at["precision"] > 0.5 and at["recall"] > 0.3


def test_mlflow_runs_logged(trained):
    import mlflow
    from fraudplat.train_xgb import setup_mlflow
    setup_mlflow()
    runs = mlflow.search_runs(experiment_names=["fraud-detection"])
    assert len(runs) >= 2
    assert "metrics.test_pr_auc" in runs.columns
