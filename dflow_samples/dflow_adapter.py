from functools import wraps
import os
import json
import traceback
from typing import Dict, Union, List

DFLOW_DISABLE = '1' == os.getenv('DFLOW_DISABLE', None)

Artifects = Union[Dict[str, Union['Artifects', str, List[str]]], str, List[str]]

def dflow_task(fn, result_file='dflow_result.json',error_file='dflow_error.txt'):
    if DFLOW_DISABLE:
        return fn
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        try:
            ret = fn(*args, **kwargs)
            _save_json(ret.value, result_file)
            return ret
        except Exception as e:
            tb = traceback.format_exc()
            with open(error_file, 'w') as f:
                f.write(tb)
            raise e
    return wrapped_fn


def _save_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f)
