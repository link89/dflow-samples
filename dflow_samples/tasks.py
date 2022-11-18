from fire import Fire
from .dflow_adapter import Result, dflow_task


@dflow_task
def gen_files(n=10, prefix='output-', text='hello world') -> Result:
    files = []
    for i in range(n):
        file = prefix + str(i) + '.txt'
        with open(file, 'w') as f:
            f.write(text)
        files.append(file)
    return Result(artifects=files)


@dflow_task
def cat_files(files):
    for file in files:
        with open(file, 'r') as f:
            print(f.read())

if __name__ == '__main__':
    Fire(dict(
        gen_files=gen_files,
        cat_files=cat_files,
    ))