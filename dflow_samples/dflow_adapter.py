from dflow import OutputArtifact, ShellOPTemplate
from functools import wraps
import os
import json
import traceback
from typing import Dict, Union, List, TypeVar, Optional, TypedDict

DFLOW_DISABLE = '1' == os.getenv('DFLOW_DISABLE', None)

Artifects = Union[Dict[str, Union['Artifects', str, List[str]]], str, List[str]]

T = TypeVar('T')
class Result(TypedDict):
    value: Optional[T]
    artifects: Optional[Artifects]


DFLOW_VALUE_FILE = '__dflow_value_file__'
DFLOW_ERROR_FILE = '__dflow_error_file__'
DFLOW_ARTIFECTS_FILE = '__dflow_artifects_file__'


def dflow_task(fn, prefix='dflow_result'):
    if DFLOW_DISABLE:
        return fn

    value_file = prefix + '_value.json'
    error_file = prefix + '_error.json'
    artifects_file = prefix + '_artifects.json'

    setattr(fn, DFLOW_VALUE_FILE, value_file)
    setattr(fn, DFLOW_ERROR_FILE, error_file)
    setattr(fn, DFLOW_ARTIFECTS_FILE, artifects_file)

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        try:
            ret: Result = fn(*args, **kwargs)
            if ret is not None:
                _save_json(ret.value, value_file)
                _save_json(ret.artifects, artifects_file)
            return ret
        except:
            tb = traceback.format_exc()
            with open(error_file, 'w') as f:
                f.write(tb)
    return wrapped_fn


def _save_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f)
