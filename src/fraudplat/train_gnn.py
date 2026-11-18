"""Step 4: fraud-ring detection with a graph neural network + score blending.

Graph: heterogeneous relations flattened into one bipartite graph
    account --uses--> device,   account --pays_with--> card
Shared devices/cards connect accounts into communities; a fraud ring shows up as a dense
cluster of accounts around a few devices. Hub entities (used by > max_entity_degree
accounts, e.g. a generic 'Windows|chrome' fingerprint) carry no ring signal and are dropped.

Model: 2-layer GraphSAGE (PyTorch Geometric), trained semi-supervised on accounts labelled
from the *training period only*; account risk scores are published to Feast
(`account_graph_risk`) and blended with XGBoost by a logistic-regression stacker fitted on the
validation period.
"""
from __future__ import annotations

import json
import logging
import time
import warnings

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from .config import load_config, path
from .feature_store import write_view
from .metrics import evaluate
from .train_xgb import setup_mlflow

log = logging.getLogger(__name__)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from torch_geometric.nn import SAGEConv


def _logit(v):
    v = np.clip(v, 1e-6, 1 - 1e-6)
    return np.log(v / (1 - v))


class RingGNN(torch.nn.Module):
    def __init__(self, d_in, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        dims = [d_in] + [hidden] * layers
        self.convs = torch.nn.ModuleList(SAGEConv(a, b) for a, b in zip(dims[:-1], dims[1:], strict=False))
        self.head = torch.nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, ei):
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, ei)), self.dropout, self.training)
        return self.head(x).squeeze(-1)


def build_graph(df: pd.DataFrame, max_deg: int):
    """Returns node features, edge_index, account index mapping."""
    acc = df.groupby("uid").agg(
        n=("amount", "size"), amt_mean=("amount", "mean"), amt_std=("amount", "std"),
        n_dev=("device_id", "nunique"), n_card=("card_id", "nunique"),
        night=("ts", lambda s: ((s % 86400) // 3600 < 6).mean()),
        span=("ts", lambda s: (s.max() - s.min()) / 86400),
        dist=("dist1", "mean"), c1=("C1", "mean"))
    acc_ids = acc.index.to_numpy()
    a_idx = pd.Series(np.arange(len(acc_ids)), index=acc_ids)

    edges, ent_feats, offset = [], [], len(acc_ids)
    for col, t in (("device_id", 0), ("card_id", 1)):
        pairs = df.loc[df[col] != "d_unknown", ["uid", col]]
        cnt = pairs.groupby(col).size()
        pairs = pairs.drop_duplicates()
        deg = pairs.groupby(col).size()
        keep = deg[(deg >= 2) & (deg <= max_deg)].index           # only entities that link accounts
        pairs = pairs[pairs[col].isin(keep)]
        e_idx = pd.Series(np.arange(len(keep)) + offset, index=keep)
        edges.append(np.stack([a_idx[pairs.uid].to_numpy(), e_idx[pairs[col]].to_numpy()]))
        f = np.zeros((len(keep), 4), dtype=np.float32)
        f[:, t] = 1
        f[:, 2] = np.log1p(deg[keep].to_numpy())
        f[:, 3] = np.log1p(cnt[keep].to_numpy())
        ent_feats.append(f)
        offset += len(keep)

    af = np.column_stack([np.log1p(acc.n), np.log1p(acc.amt_mean), np.log1p(acc.amt_std.fillna(0)),
                          acc.n_dev, acc.n_card, acc.night, np.log1p(acc.span), np.log1p(acc.dist.fillna(0)),
                          np.log1p(acc.c1.fillna(0))]).astype(np.float32)
    af = (af - af.mean(0)) / (af.std(0) + 1e-6)
    d_acc, d_ent = af.shape[1], 4
    X = np.zeros((offset, d_acc + d_ent + 1), dtype=np.float32)
    X[: len(acc_ids), :d_acc] = af
    X[: len(acc_ids), -1] = 1                                         # node-type flag: account
    X[len(acc_ids):, d_acc:d_acc + d_ent] = np.vstack(ent_feats) if ent_feats else 0
    e = np.hstack(edges) if edges else np.zeros((2, 0), dtype=int)
    ei = np.hstack([e, e[::-1]])                                      # undirected
    stats = {"n_accounts": int(len(acc_ids)), "n_entity_nodes": int(offset - len(acc_ids)),
             "n_edges": int(e.shape[1])}
    return torch.tensor(X), torch.tensor(ei, dtype=torch.long), a_idx, stats


def run(table: pd.DataFrame | None = None) -> dict:
    cfg = load_config()
    gcfg = cfg["gnn"]
    tp = cfg["serving"]["target_precision"]
    art = path(cfg["serving"]["artifacts_dir"])
    if table is None:
        table = pd.read_parquet(path(cfg["data"]["processed_dir"]) / "training_table.parquet")
    torch.manual_seed(cfg["data"]["seed"])
    t0 = time.time()
    X, ei, a_idx, gstats = build_graph(table, gcfg["max_entity_degree"])

    # labels: account is fraudulent if it had fraud in the TRAINING period (no future labels)
    tr = table[table.split == "train"]
    lab = tr.groupby("uid").is_fraud.max()
    nodes = a_idx[lab.index].to_numpy()
    y = torch.tensor(lab.to_numpy(), dtype=torch.float32)
    perm = np.random.default_rng(0).permutation(len(nodes))
    cut = int(0.85 * len(nodes))
    tr_n, va_n = torch.tensor(nodes[perm[:cut]]), torch.tensor(nodes[perm[cut:]])
    y_tr, y_va = y[perm[:cut]], y[perm[cut:]]
    pos_w = torch.tensor((y_tr == 0).sum().item() / max((y_tr == 1).sum().item(), 1))

    # "Label trick" (Wang et al., 2021): known labels enter as input features of a random half
    # of the labelled accounts each epoch while the loss is computed on the other half, so the
    # model learns guilt-by-association (device/card shared with confirmed fraudsters) without
    # ever seeing the label of the node it is predicting.
    n_nodes = X.shape[0]

    def with_labels(input_nodes, input_y):
        lab_feat = torch.zeros((n_nodes, 2))
        lab_feat[input_nodes, 0] = 1.0                                  # label known
        lab_feat[input_nodes, 1] = input_y                              # label value
        return torch.cat([X, lab_feat], 1)

    X_eval = with_labels(tr_n, y_tr)                                    # all training labels visible
    with warnings.catch_warnings():               # PyG emits a Python 3.13 typing deprecation notice
        warnings.simplefilter("ignore")
        model = RingGNN(X.shape[1] + 2, gcfg["hidden"], gcfg["layers"], gcfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=gcfg["lr"], weight_decay=1e-4)
    best, best_state, patience = -1, None, 0
    gen = torch.Generator().manual_seed(0)
    for ep in range(gcfg["epochs"]):
        model.train()
        m = torch.rand(len(tr_n), generator=gen) < 0.5
        opt.zero_grad()
        out = model(with_labels(tr_n[m], y_tr[m]), ei)
        loss = F.binary_cross_entropy_with_logits(out[tr_n[~m]], y_tr[~m], pos_weight=pos_w)
        loss.backward()
        opt.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                ap = average_precision_score(y_va.numpy(), torch.sigmoid(model(X_eval, ei)[va_n]).numpy())
            if ap > best:
                best, best_state, patience = ap, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 10:
                    break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():                                               # inference: every train-period label
        acc_score = torch.sigmoid(model(with_labels(torch.tensor(nodes), y), ei)).numpy()[: len(a_idx)]
    scores = pd.Series(acc_score, index=a_idx.index, name="gnn_score")
    fit_s = time.time() - t0
    torch.save(model.state_dict(), art / "gnn_model.pt")

    # Device-level ring risk = mean / max account risk over the accounts seen on the device.
    # An account takeover from a ring device has a clean *account* but a hot *device*.
    pairs = table.loc[table.device_id != "d_unknown", ["uid", "device_id"]].drop_duplicates()
    pairs["s"] = pairs.uid.map(scores)
    dev_risk = pairs.groupby("device_id").s.agg(gnn_dev_mean="mean", gnn_dev_max="max")

    # publish to Feast (online serving reads it; snapshot time = end of observed data)
    snap_ts = pd.to_datetime(int(table.ts.max()), unit="s", utc=True)
    write_view("account_graph_risk", pd.DataFrame({"uid": scores.index, "event_timestamp": snap_ts,
                                                   "gnn_score": scores.to_numpy().astype("float32")}))
    write_view("device_graph_risk", pd.DataFrame({"device_id": dev_risk.index, "event_timestamp": snap_ts,
                                                  "gnn_dev_mean": dev_risk.gnn_dev_mean.astype("float32").to_numpy(),
                                                  "gnn_dev_max": dev_risk.gnn_dev_max.astype("float32").to_numpy()}))

    # ---------- blend with XGBoost (logistic-regression stacker fitted on the validation period) ----------
    default = float(scores.median())
    pv = pd.read_parquet(art / "pred_valid.parquet")
    pt = pd.read_parquet(art / "pred_test.parquet")
    dev_of = table.set_index("transaction_id").device_id
    for p in (pv, pt):
        dev = p.transaction_id.map(dev_of)
        p["p_gnn"] = p.uid.map(scores).fillna(default)
        p["gnn_dev_mean"] = dev.map(dev_risk.gnn_dev_mean).fillna(default)
        p["gnn_dev_max"] = dev.map(dev_risk.gnn_dev_max).fillna(default)

    def feats(p, cols):
        return np.column_stack([_logit(p[c].to_numpy()) for c in cols])

    full = ["p_xgb", "p_gnn", "gnn_dev_mean", "gnn_dev_max"]
    stack = LogisticRegression(C=1.0, max_iter=1000).fit(feats(pv, full), pv.y)
    blend_va = average_precision_score(pv.y, stack.predict_proba(feats(pv, full))[:, 1])
    xgb_va = average_precision_score(pv.y, pv.p_xgb)
    use_cols = full
    if blend_va < xgb_va:            # guard: never ship a blend that is worse on validation
        log.warning("Blend (%.4f) < XGBoost (%.4f) on validation; GNN weight set to 0", blend_va, xgb_va)
        use_cols = ["p_xgb"]
        stack = LogisticRegression(C=1.0, max_iter=1000).fit(feats(pv, use_cols), pv.y)
    for p in (pv, pt):
        p["p_blend"] = stack.predict_proba(feats(p, use_cols))[:, 1]
    coef = dict(zip(use_cols, map(float, stack.coef_[0]), strict=True))

    seen = set(lab.index)
    new_acc = ~pt.uid.isin(seen)
    res = {
        "graph": gstats,
        "gnn_backend": "torch_geometric.SAGEConv",
        "gnn_account_val_pr_auc": float(best),
        "fit_seconds": round(fit_s, 1),
        "blend": {"features": use_cols, "coef": coef, "intercept": float(stack.intercept_[0]),
                  "valid_pr_auc_xgb": float(xgb_va), "valid_pr_auc_blend": float(blend_va)},
        "test": {
            "xgb": evaluate(pt.y, pt.p_xgb, tp),
            "gnn": evaluate(pt.y, pt.p_gnn, tp),
            "blend": evaluate(pt.y, pt.p_blend, tp),
        },
        "test_new_accounts_only": {
            "n_txn": int(new_acc.sum()),
            "xgb_pr_auc": float(average_precision_score(pt.y[new_acc], pt.p_xgb[new_acc])) if pt.y[new_acc].sum() else None,
            "blend_pr_auc": float(average_precision_score(pt.y[new_acc], pt.p_blend[new_acc])) if pt.y[new_acc].sum() else None,
        },
    }
    # threshold: chosen on VALIDATION for the target precision, then reported on test
    vm = evaluate(pv.y, pv.p_blend, tp)
    res["threshold"] = vm["threshold_at_target_precision"]
    thr = res["threshold"]
    flag = pt.p_blend >= thr
    res["test_at_threshold"] = {
        "threshold": thr, "alerts": int(flag.sum()), "alert_rate_pct": round(100 * flag.mean(), 3),
        "precision": float(pt.y[flag].mean()) if flag.any() else 0.0,
        "recall": float(pt.y[flag].sum() / max(pt.y.sum(), 1)),
    }
    (art / "blend.json").write_text(json.dumps({**res["blend"], "threshold": thr, "gnn_default": default}, indent=2))
    (art / "gnn_summary.json").write_text(json.dumps(res, indent=2))
    pt.to_parquet(art / "pred_test.parquet")

    setup_mlflow()
    with mlflow.start_run(run_name="gnn_blend"):
        mlflow.log_params({**gcfg, **gstats})
        mlflow.log_metric("gnn_account_val_pr_auc", best)
        for m, d in res["test"].items():
            mlflow.log_metrics({f"test_{m}_{k}": v for k, v in d.items()})
        mlflow.log_artifact(str(art / "gnn_model.pt"))
        mlflow.log_dict(res, "gnn_summary.json")
    log.info("GNN+blend: test PR-AUC xgb %.4f -> blend %.4f", res["test"]["xgb"]["pr_auc"], res["test"]["blend"]["pr_auc"])
    return res
