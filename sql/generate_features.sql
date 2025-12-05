TRUNCATE TABLE feature_zone.inventory_features;

INSERT INTO feature_zone.inventory_features
WITH top10 AS (
    SELECT sku_id
    FROM clean_zone.inventory_transactions
    GROUP BY sku_id
    ORDER BY sum(quantity_consumed) DESC
    LIMIT 10
),

daily AS (
    SELECT
        sku_id,
        toDate(transaction_time) AS date,
        sum(quantity_consumed) AS daily_qty
    FROM clean_zone.inventory_transactions
    WHERE sku_id IN (SELECT sku_id FROM top10)
    GROUP BY sku_id, date
),

daily_features AS (
    SELECT
        sku_id,
        date,
        daily_qty,
        lagInFrame(daily_qty, 1)
            OVER (PARTITION BY sku_id ORDER BY date ASC) AS lag_1,
        sum(daily_qty)
            OVER (PARTITION BY sku_id ORDER BY date ASC ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS qty_last_7_days,
        sum(daily_qty)
            OVER (PARTITION BY sku_id ORDER BY date ASC ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS qty_last_30_days,
        avg(daily_qty)
            OVER (PARTITION BY sku_id ORDER BY date ASC ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS avg_last_7,
        avg(daily_qty)
            OVER (PARTITION BY sku_id ORDER BY date ASC ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS avg_last_30
    FROM daily
),

events AS (
    SELECT
        t.sku_id,
        t.transaction_time AS feature_timestamp,
        t.quantity_consumed,
        df.lag_1,
        df.qty_last_7_days,
        df.qty_last_30_days,
        df.avg_last_7,
        df.avg_last_30
    FROM clean_zone.inventory_transactions AS t
    INNER JOIN top10 AS tp
        ON t.sku_id = tp.sku_id
    LEFT JOIN daily_features AS df
        ON t.sku_id = df.sku_id
       AND toDate(t.transaction_time) = df.date
)

SELECT
    sku_id,
    feature_timestamp,
    quantity_consumed,
    COALESCE(lag_1, 0),
    COALESCE(qty_last_7_days, 0),
    COALESCE(qty_last_30_days, 0),
    COALESCE(avg_last_7, 0),
    COALESCE(avg_last_30, 0)
FROM events
ORDER BY sku_id, feature_timestamp;

