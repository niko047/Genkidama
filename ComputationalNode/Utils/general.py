from ast import literal_eval
from schema import Schema, And, Use, Optional, SchemaError
import pandas as pd
import numpy as np
from io import BytesIO
import codecs
import pickle

payload_schema = Schema({
    "metadata": {
        "worker_node": str,
        "inquiring_node": str,
        "start_time": str,
        "max_cpu_allowed": float,
        "timeout_boundary": int,
        "task_type": str,
    },
    "data": {
        "X": str,
        "y": str,
    },
    "instructions": {
        "code" : str,
        "param_grid" : dict,
        "algorithm_name": str,
    },
})

def is_valid_input_payload(input_payload: str) -> dict:
    if not isinstance(input_payload, str):
        raise Exception('Input payload is not a valid string')

    try:
        dict_input_payload = literal_eval(input_payload)
    except:
        Exception('Input payload cannot be read as a dict')

    payload_schema.validate(dict_input_payload)

    return dict_input_payload


def str_bytes_to_pandas_df(str_bytes: str) -> pd.DataFrame:
    df = pickle.loads(codecs.decode(str_bytes.encode(), "base64"))
    return df

