# dflow-samples

基于 dflow 和 Bohrium 的工作流项目示例。

## 使用

### 在本地执行

使用前需安装依赖如下：

```bash
poetry install
pip install tensorflow==2.10.1  # 如已通过conda或其它途径安装可省略此步
```

然后即可通过以下命令调用实现的方法 (其中训练所用的 outcar 文件和预测所用的轨迹文件均为预先生成)

```bash
python -m dflow_samples.main train_model --elements=[Na] --outcar_folders="['./data/nmr/p6322', './nmr/data/train/p63mcm']"

python -m dflow_samples.main predict --elements=[Na] --traj_path=./data/nmr/predict_fcshifts_example.xyz --model=./out/model 
```

### 通过 Docker 执行

为支持在 dflow 执行，该项目提供了Dockerfile用于构建用于代码执行的容器，使用前先使用以下命令确保容器被正确构建

```bash
docker build -t dflow-nmr .
```

构建完成后可通过以下命令执行（注意需要将外部数据目录挂载到内部, 输出模型也需指定到挂载目录上）

```bash
 docker run -v $PWD/data/nmr:/data dflow-nmr python -m dflow_samples.main train_model --elements=[Na] --outcar_folders="['/data/train/p6322', '/data/train/p63mcm']" --out_dir /data/out

 docker run -v $PWD/data/nmr:/data dflow-nmr python -m dflow_samples.main predict --elements=[Na] --traj_path=/data/predict_fcshifts_example.xyz --model=/data/out/model
```

### 通过 Singularity 执行

为支持在 Bohrium 平台运行，首先需要将 docker 镜像转换为 singularity 镜像并上传到嘉庚超算的容器目录中。

在本地构建镜像可以使用 conda 安装 singularity的运行时:

```bash
conda install -c conda-forge singularity
```

singularity镜像可直接从 docker 镜像中转化

```bash
singularity build dflow_nmr.sif docker-daemon://dflow-nmr:latest
```

构建完成后可使用以下命令执行

```bash
singularity exec --bind ./data/nmr:/data dflow_nmr.sif python -m dflow_samples.main train_model --elements=[Na] --outcar_folders="['/data/train/p6322', '/data/train/p63mcm']" --out_dir /data/out

singularity exec --bind ./data/nmr:/data dflow_nmr.sif python -m dflow_samples.main predict --elements=[Na] --traj_path=/data/predict_fcshifts_example.xyz --model=/data/out/model
```

### 在 Bohrium 平台上执行

为了能够在 Bohrium 平台上使用嘉庚超算执行相关代码，需要将上一步生成的 sif 镜像部署到超算上的指定位置。

部署完成后可参考此 [教程](./docs/tutorial.md) 去使用。
