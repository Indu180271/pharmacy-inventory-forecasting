CREATE DATABASE IF NOT EXISTS feature_zone;

CREATE TABLE IF NOT EXISTS feature_zone.inventory_features
(
    sku_id Int32,
    feature_timestamp DateTime,
    quantity_consumed Float32,
    lag_1 Float32,
    qty_last_7_days Float32,
    qty_last_30_days Float32,
    avg_last_7 Float32,
    avg_last_30 Float32
)
ENGINE = MergeTree()
ORDER BY (sku_id, feature_timestamp);

