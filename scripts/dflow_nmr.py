from dflow import ShellOPTemplate
from dflow import InputParameter, InputArtifact, OutputParameter, OutputArtifact
from dflow import Step, Workflow
from dflow import upload_artifact
from dflow.plugins.dispatcher import DispatcherExecutor
from typing import List
import shlex
import json
import getpass

container_image = "registry.dp.tech/dptech/dflow-nmr:v0.1.2"

# Step 1: train model
nmr_train_script = ' '.join([
    "python -m dflow_samples.main train_model",
    "--elements={{inputs.parameters.elements}}",
    "--outcar_folders_dir=/tmp/data/train",
    "--out_dir /tmp/out"
])
print(nmr_train_script)
nmr_train_template = ShellOPTemplate(
    name="nrm-train",
    image=container_image,
    script=nmr_train_script,
)
nmr_train_template.inputs.parameters = {"elements": InputParameter()}
nmr_train_template.inputs.artifacts = {"data": InputArtifact(path="/tmp/data")}
nmr_train_template.outputs.artifacts = {"out": OutputArtifact(path="/tmp/out")}


# Step 2: predict
nmr_predict_script = ' '.join([
    "python -m dflow_samples.main predict",
    "--elements={{inputs.parameters.elements}}",
    "--traj_path=/tmp/data/predict_fcshifts_example.xyz",
    "--model=/tmp/out/model"
])
print(nmr_predict_script)
nmr_predict_template = ShellOPTemplate(
    name="nrm-predict",
    image=container_image,
    script=nmr_predict_script,
)
nmr_predict_template.inputs.parameters = {"elements": InputParameter()}
nmr_predict_template.inputs.artifacts = {
    "data": InputArtifact(path="/tmp/data"),
    "out": InputArtifact(path="/tmp/out"),
}
nmr_train_template.outputs.parameters = {
    "msg": OutputParameter(value_from_path="./dflow_result.json")}

## Build workflow

def run_nmr_workflow(elements: List[str], data: str, executor):
    wf = Workflow(name="nmr-workflow")

    data_artifact = upload_artifact(data)
    quoted_elements = shlex.quote(json.dumps(elements))  # list args needs to be quoted due to fire

    nmr_train = Step(
        name="nmr-train",
        template=nmr_train_template,
        parameters={"elements": quoted_elements},
        artifacts={"data": data_artifact},
        executor=executor,
    )
    wf.add(nmr_train)

    nmr_predict = Step(
        name="nmr-predict",
        template=nmr_predict_template,
        parameters={"elements": quoted_elements},
        artifacts={"data": data_artifact, 'out': nmr_train.outputs.artifacts['out']},
        executor=executor,
    )
    wf.add(nmr_predict)
    wf.submit()


## Run workflow
if __name__ == '__main__':
    dispatcher_executor = DispatcherExecutor(
        machine_dict={
            "batch_type": "Bohrium",
            "context_type": "Bohrium",
            "remote_profile": {
                "email": "xuweihong.cn@xmu.edu.cn",
                "password": getpass.getpass('password:'),
                "program_id": 11035,
                "ondemand": 1,
                "input_data": {
                    "job_type": "container",
                    "platform": "ikkemhpc",
                    "machine_type": "c64_m128_cpu",
                },
            },
        },
    )
    run_nmr_workflow(elements=['Na'], data='./data/nmr', executor=dispatcher_executor)
