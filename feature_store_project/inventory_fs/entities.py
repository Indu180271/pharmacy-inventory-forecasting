from feast import Entity
from feast.types import String

sku = Entity(
    name="sku_id",
    value_type=String,
    description="Unique SKU identifier",
)
