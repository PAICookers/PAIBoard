# PAIBoard

PAIBoard是用于PAICORE应用部署的类，基类是PAIBoard_base定义了通用的基本函数，介绍如下。在此类基础上派生出来的类都需要定义收发帧的函数，如以太网平台需要使用socket进行帧的收发，PCIe平台需要使用XDMA驱动来进行帧的收发等等。

## PAIBoard_base

1、生成gen_init_frame

    init操作相当于膜电位复位，对于目前的芯片来说，需要用配置帧配合init和work帧进行init操作，并且每个core都要init。
    目前通过读取core_params.json来获取所有core坐标，或许可以用core*实现。

2、gen_sync_frame与运行的timestep和网络层数有关

3、input_frames_info生成，读取input_proj_info.json

    对于单个输入节点，可以知道输入对应每个core和addr_axon，然后生成相应的前56bit帧头数组，即为input_frames_info，然后推理生成脉冲帧时对非零数据进行拼接即可。

    多个输入节点生成列表/字典形式的input_frames_info，输入列表/字典，生成对应的脉冲帧，然后所有脉冲帧合并输入到芯片即可，

    对于batch推理模式，需要复制batch份的input_frames_info，其中的tick_relative信息则递增生成。

    todo:还需要要做输入扇入测试

4、output_frames_info生成，读取output_dest_info.json

    对于单个输出节点，可以知道每个输出的addr_axon地址，然后生成相应的前56bit帧头数组，即为output_frames_info，然后获取芯片输出后，按照特定的解码函数进行解码(排序后找对应位置的数据)，得到numpy数组。

    对于多个输出节点，生成字典形式的output_frames_info，芯片输出帧时并没有办法将不同节点的输出分开输出，因此需要借助output_frames_info中的信息将其输出到字典中，需要用字典来输出，列表是无序的，无法判断输出。

    对于batch模式，同样是复制batch份，递增生成tick_relative

    todo:工具链输出不支持扇入？less1152，目前只支持少于1152个输出的结构。

5、config

    直接将工具链生成的配置帧送入芯片即可

6、推理__call__

生成输入帧genSpikeFrame(input_frames_info,ifi)，分为四种情况

    ·单输入，非batch模式
        直接输入np格式的spike的对应的ifi即可

    ·单输入，batch模式
        将输入spike按照timestep的维度进行堆叠送入。batch模式首先要指定当前推理的最大batch，然后并非每次推理都会用到最大batch，因此还需要实时判断当前输入的脉冲维度对应多少个batch，生成对应数目的sync_frame和ifi。ifi数目需要与spike数目对应上才能正确地生成脉冲帧

    ·多输入，非batch模式
        输入以字典的方式送入，然后生成的脉冲帧可以无序，只要对应的输入和ifi对上即可，可用输入节点名作为key

    ·多输入，batch模式(todo)
        目前认为多个输入节点输入的batch是一致的？
        不一致也能处理？会有什么后果？

送入芯片
    将initframe、spikeframe、syncframe组合送入芯片即可

解析输出帧，获得芯片的输出帧后进行解析，获得np数组genOutputSpike，分为四种情况
    ·单输出，非batch模式
        输入为脉冲帧和ofi，得到填充了零的数组，因为芯片不会输出非零的数据，所以没有输出的"位置"，默认填充0。

    ·单输出，batch模式
        根据输入时的batch情况，取对应数量的ofi进行解析，得到的输出维度将会在timestep维度进行堆叠batch份，需要后续进行拆分。

    ·多输出，非batch模式
        decode_spike_less1152支持多个oframe_infos为输入，然后使用for循环逐个将字典中的结果读取出来。
    
    ·多输出，batch模式(todo)
            
由于扇入不足的问题，这里实际上还做了拆分推理的处理split_inference，比如一次推理需要128的timesteps，可以先推理32个timesteps，然后收集输出后不复位，再送入32个timesteps的数据，重复四次后就可以收到完整的推理结果，这是因为芯片和工具链的限制做出的妥协，在一定程度上影响推理性能。

## runtime实现说明

需与工具链方了解此部分实现

## batch模式与多输入输出模式

batch与原本的网络推理不同之处在于，可以在就是可以批量送入数据进行推理，利用芯片流水线推理，使得芯片的推理性能大幅提升。batch模式实际上影响的是一次性输入网络的**数据**数目，本来timestep为4的网络，只要把4个timestep的数据都送入芯片的timeslot中即可。如果使用batch模式进行推理，则可以将芯片的256个timeslot都填满，对于timestep为4的应用，就可以填入64组数据，然后芯片也会一次性输出64组输出。需要注意的是，每次推理之间没办法实现芯片的复位/init，因此需要在算理方面先验证好，即使膜电位不复位也不影响推理结果。

多输入输出则是比较朴素，芯片本身就可以接收多组输入，只需要配置好输入帧即可。

## PAIBoard_SIM

使用PAICORE运行模拟器搭建的运行单元，能在一定程度上验证帧的编解码是否运行正确。

## PAIBoard_Ethernet

使用DMA_Ethernet类作为与板卡以太网通信的平台，双方都使用socket进行通信。其主要实现是定义了一系列的通信指令，在FPGA端收到相应的指令后，控制硬件进行收发帧，寄存器配置等一系列的任务。

服务端的程序主要参考PAIBoard/paiboard/ethernet/server/server_for_array.py，类似状态机的运行，根据指令完成收发帧、寄存器配置、串口复位等硬件的操作。

而本地的PAIBoard作为客户端仅作为数据准备和指令/数据发送方。