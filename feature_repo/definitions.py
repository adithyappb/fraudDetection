"""Feast feature definitions. Offline source = parquet snapshots written by the pipeline;
online store = SQLite (swap for Redis in production via feature_store.yaml)."""
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "feast"

account = Entity(name="account", join_keys=["uid"], value_type=ValueType.STRING,
                 description="IEEE-CIS client key: card1 + addr1 + account-start day")
device = Entity(name="device", join_keys=["device_id"], value_type=ValueType.STRING,
                description="Device fingerprint: DeviceInfo | browser | screen")


def _src(name):
    return FileSource(name=f"{name}_src", path=str(DATA / f"{name}.parquet"),
                      timestamp_field="event_timestamp", created_timestamp_column="created")


account_profile = FeatureView(
    name="account_profile", entities=[account], ttl=timedelta(days=400), online=True,
    schema=[Field(name=n, dtype=Float32) for n in
            ["acct_txn_count", "acct_amt_mean", "acct_amt_std", "acct_n_devices",
             "acct_night_ratio", "acct_active_days"]],
    source=_src("account_profile"), tags={"owner": "fraud-ml", "freshness": "daily"},
)

device_profile = FeatureView(
    name="device_profile", entities=[device], ttl=timedelta(days=400), online=True,
    schema=[Field(name=n, dtype=Float32) for n in ["dev_n_accounts", "dev_txn_count", "dev_amt_mean"]],
    source=_src("device_profile"), tags={"owner": "fraud-ml", "freshness": "daily"},
)

account_graph_risk = FeatureView(
    name="account_graph_risk", entities=[account], ttl=timedelta(days=400), online=True,
    schema=[Field(name="gnn_score", dtype=Float32)],
    source=_src("account_graph_risk"), tags={"owner": "fraud-ml", "producer": "train_gnn"},
)

device_graph_risk = FeatureView(
    name="device_graph_risk", entities=[device], ttl=timedelta(days=400), online=True,
    schema=[Field(name="gnn_dev_mean", dtype=Float32), Field(name="gnn_dev_max", dtype=Float32)],
    source=_src("device_graph_risk"), tags={"owner": "fraud-ml", "producer": "train_gnn"},
)

ALL_OBJECTS = [account, device, account_profile, device_profile, account_graph_risk, device_graph_risk]
