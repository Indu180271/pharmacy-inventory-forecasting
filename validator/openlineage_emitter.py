# openlineage_emitter.py
import os
from datetime import datetime
import pandas as pd

try:
    from openlineage.client import OpenLineageClient
    from openlineage.client.run import RunEvent, Run, Job, Dataset, RunState
    from openlineage.client.facet import SchemaDatasetFacet, SchemaField
    OPENLINEAGE = True
except Exception:
    OPENLINEAGE = False

OL_URL = os.getenv("OPENLINEAGE_URL", "http://marquez:8080/api/v1/lineage")
OL_NAMESPACE = os.getenv("OPENLINEAGE_NAMESPACE", "inventory_pipeline")


def _df_to_schema_facet(df: pd.DataFrame):
    fields = [SchemaField(name=str(c), type="STRING") for c in df.columns]
    return SchemaDatasetFacet(fields=fields)


def emit_event(job_name: str,
               run_id: str,
               state: str = "START",
               inputs: list = None,
               outputs: list = None,
               df: pd.DataFrame = None):

    if not OPENLINEAGE:
        return

    try:
        client = OpenLineageClient(OL_URL)

        ol_inputs, ol_outputs = [], []

        # ---------------------
        # Build Inputs
        # ---------------------
        if inputs:
            for i, inp in enumerate(inputs):
                facets = {}
                if df is not None and i == 0:
                    facets = {"schema": _df_to_schema_facet(df)}

                ol_inputs.append(
                    Dataset(
                        namespace=inp.get("namespace", "source"),
                        name=inp["name"],
                        facets=facets
                    )
                )

        # ---------------------
        # Build Outputs
        # ---------------------
        if outputs:
            for out in outputs:
                ol_outputs.append(
                    Dataset(
                        namespace=out.get("namespace", OL_NAMESPACE),
                        name=out["name"]
                    )
                )

        # ---------------------
        # State Mapping
        # ---------------------
        state_map = {
            "START": RunState.START,
            "COMPLETE": RunState.COMPLETE,
            "FAIL": RunState.FAIL
        }

        # ---------------------
        # Build OpenLineage Event
        # ---------------------
        ev = RunEvent(
            eventType=state_map.get(state, RunState.START),
            eventTime=datetime.utcnow().isoformat(),
            run=Run(runId=run_id),
            job=Job(namespace=OL_NAMESPACE, name=job_name),
            inputs=ol_inputs,
            outputs=ol_outputs,
            producer="gx_pipeline"
        )

        client.emit(ev)

    except Exception as e:
        print(f"[openlineage_emitter] emit failed for {job_name} {state}: {e}")

