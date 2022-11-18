from collections import namedtuple
from fire import Fire
from typing import Dict, Union, List, Tuple, Any, TypeVar, Optional


# Artifects should be string, list of string or string dict of artifects
Artifects = Union[Dict[str, Union['Artifects', str, List[str]]], str, List[str]]

T = TypeVar('T')
# Result is a tumple of (value, artifects), value must be serializable
Result = Tuple[Optional[T], Artifects]


def gen_files(n=10, prefix='output-', text='hello world') -> Result:
    files = []
    for i in range(n):
        file = prefix + str(i) + '.txt'
        with open(file, 'w') as f:
            f.write(text)
        files.append(file)
    return None, files


def cat_files(files):
    for file in files:
        with open(file, 'r') as f:
            print(f.read())

if __name__ == '__main__':
    Fire(dict(
        gen_files=gen_files,
        cat_files=cat_files,
    ))