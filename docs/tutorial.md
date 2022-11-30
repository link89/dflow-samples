# 核磁预测工作流上机操作步骤

## 简介
该工作流利用Bohrium平台上的控制节点部署 argo workflow 执行dflow 脚本, 脚本通过 dflow 提供的dispatcher使用嘉庚超算中心预部署的容器完成相关的计算工作。

## 步骤

### 创建节点并运行 Argo Workflow

参见：https://dptechnology.feishu.cn/docx/XZcMdb1P4ocovqxAxYkcCZg2nwc

### 运行 Jupiter Notebook

```bash

## 创建工作目录
mkdir -p dflow_nmr
cd dflow_nmr

## 准备数据
tar -xvf ../nmr.tar.bz2

## 检查数据
tree -d 

# 正确解压后应有如下的目录结构
# .
# └── nmr
#     └── train
#         ├── p6322
#         └── p63mcm

## 启动 Jupyter Notebook
jupyter notebook
```

### 运行 dflow 脚本

在 Jupiter Notebook 中新建一个 notebook, 将 [dflow_nmr.py](../scripts/dflow_nmr.py) 代码复制到一个 cell 当中，运行该 cell 即可。

cell 运行后会要求输入对应的 Bohrium 平台的邮箱和密码，作业提交成功后会看到如下输出

```txt
Workflow has been submitted (ID: nmr-workflow-bbmn5, UID: 906e6d3a-ca14-4c96-bc42-f04460b209a9)
```

此时可打开 https://127.0.0.1:2746/workflows 在 Argo Workflow 中查看对应的作业执行情况。
