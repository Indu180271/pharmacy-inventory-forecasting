from clickhouse_driver import Client
import pandas as pd
from src.config.settings import settings

class ClickHouseClient:

    def __init__(self):
        self.client = Client(
            host=settings.CLICKHOUSE_HOST,
            port=settings.CLICKHOUSE_PORT,
            user=settings.CLICKHOUSE_USER,
            password=settings.CLICKHOUSE_PASSWORD,
            database=settings.CLICKHOUSE_DATABASE
        )

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a ClickHouse query and return a pandas DataFrame.
        """
        # Execute query ? returns list of tuples
        rows, meta = self.client.execute(query, with_column_types=True)

        # Fetch column names
        columns = [col[0] for col in meta]

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=columns)

        return df


clickhouse_client = ClickHouseClient()
