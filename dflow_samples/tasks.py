from collections import namedtuple
from fire import Fire
from typing import Dict, Union, List, Tuple, Any, TypeVar, Optional

T = TypeVar('T')

Artifects = Union[Dict[str, Union[str, 'Artifects', List[str]]], List[str]]
Result = Tuple[Optional[T], Artifects]


def gen_files(n=10, prefix='output-', text='hello world') -> Result:
    output_files = []
    for i in range(n):
        file = prefix + str(i) + '.txt'
        with open(file, 'w') as f:
            f.write(text)
        output_files.append(file)
    return None, dict(output_files=output_files)


def cat_files(files):
    for file in files:
        with open(file, 'r') as f:
            print(f.read())


if __name__ == '__main__':
    Fire(dict(
        gen_files=gen_files,
        cat_files=cat_files,
    ))