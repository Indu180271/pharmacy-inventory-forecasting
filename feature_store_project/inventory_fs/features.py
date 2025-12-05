from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Int64, Float64, String
from entities import sku
from feast.infra.offline_stores.contrib.clickhouse_offline_store.clickhouse_source import ClickhouseSource

inventory_source = ClickhouseSource(
    database="feature_zone",
    table="inventory_features",
    timestamp_field="feature_timestamp",
)

inventory_features = FeatureView(
    name="inventory_features",
    entities=[sku],
    ttl=timedelta(days=90),
    schema=[
        Field(name="quantity_consumed", dtype=Int64),
        Field(name="lag_1", dtype=Int64),
        Field(name="qty_last_7_days", dtype=Int64),
        Field(name="qty_last_30_days", dtype=Int64),
        Field(name="avg_last_7", dtype=Float64),
        Field(name="avg_last_30", dtype=Float64),
    ],
    online=True,
    batch_source=inventory_source,
)
