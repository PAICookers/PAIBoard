# PAIBoard

# 概述
基于PAICORE实现的模拟器与上板部署终端程序，既能用于PAICORE上板前的模拟，也能支持上板应用程序的运行。目前已经实现针对每个平台可以例化不同的PAIBoard类，实现在不同平台的部署

三个平台包括:

    PAIBoard_SIM        : 软件仿真平台,无需硬件实物就可本地运行
    PAIBoard_PCIe       : PCIe平台
    PAIBoard_Ethernet   : 7100对应的以太网平台

## 简介


### 工具链
无论是要做哪件事，都需要先使用**工具链**产生相应的输出结果和帧信息，即如下四个文件
(config_cores_all.bin、core_params.json、input_proj_info.json、output_dest_info.json)
目前支持的工具链为PAIBOX
- https://github.com/PAICookers/PAIBox

### 应用实例
本项目中提供了3个应用程序参考(在example目录下)

    01_bypass_net.py
    02_bypass_net_2layer.py
    03_mnist_1layer.py

    bypass_net是输入和输出完全一致的网络，因为阈值是1，权重是单位矩阵，复位电平是0
    mnist则是实现了完整推理的流程


使用本项目上板前需参考上述的example样例，编写自己的应用程序，主要包括**读取数据集、数据的脉冲编码、输入脉冲数据给snn进行推理、得到推理输出的脉冲**。

### PAIBoard使用
    例化一个PAIBoard对象(三个平台任选其一)，需要用户填写工具链的结果存放目录baseDir、应用的timestep以及网络的层数layer_num，进行配置config后就可以输入数据进行推理，config可填写应用输出的帧数预估，如果不知道填10000。

    baseDir  :存放了所有工具链导出数据的文件夹
    timestep :SNN推理所需的时间步
    layer_num:SNN网络结构的层数,或是输入节点到输出节点经过的神经元数目


    # PAIBoard例化
    snn = PAIBoard_SIM(baseDir, timestep, layer_num=0)
    snn.config(oFrmNum=10000)

    # 推理实例
    # 根据你的应用产生相应的输入脉冲,送入snn后推理得到输出脉冲。
    output_spike = snn(input_spike)

极力推荐用户先使用PAIBoard_SIM上板模拟器来进行测试，无需硬件即可进行调试。

## 实例复现
### 上位机运行环境
以conda为例

    conda create -n PAIBOX python=3.11
    conda activate PAIBOX

    pip install paicorelib numpy tqdm
    pip install pyserial

### 上板流程

#### 模拟器测试
    直接在上位机运行应用程序，应用程序中需例化PAIBoard_SIM
    
#### PCIe板卡
    FPGA与PAICORE板卡上电，连接PCIe线
    直接在上位机运行应用程序，应用程序中需例化PAIBoard_PCIe

#### Ethernet板卡
FPGA与PAICORE板卡上电，连接网线，新开一个终端，命令行输入以下指令(pwd:xilinx)：

    ssh xilinx@192.168.31.100   
    cd jupyter_notebooks/PAICORE/Ethernet/
    sudo python3 server.py
显示Wating...便可在上位机运行应用程序，应用程序中需例化PAIBoard_Ethernet

### 样例数据下载，数据放到PAIBoard/result目录下

    https://disk.pku.edu.cn/link/AAD7D8BA3609BF4515A01E6842F6D7E589

### 运行
vscode打开PAIBoard文件夹，点开某个py文件直接右上角运行即可
