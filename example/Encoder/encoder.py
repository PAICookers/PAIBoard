import numpy as np
from example.Encoder.conv_acc import ConvolutionalLayer
from paiboard.utils.timeMeasure import time_calc_addText, get_original_function


@time_calc_addText("PoissonEncoder")
def PoissonEncoder(x, timesteps):
    assert timesteps >= 1
    np.random.seed(0)
    x = np.expand_dims(x, axis=0).repeat(timesteps, axis=0)
    spike_out = np.less_equal(np.random.randint(0, 255, x.shape), x).astype(np.uint8)
    return spike_out


def PoissonEncoderWrap(x, timesteps, TimeMeasure):
    if TimeMeasure:
        return PoissonEncoder(x, timesteps)
    else:
        original_PoissonEncoder = get_original_function(PoissonEncoder)
        return original_PoissonEncoder(x, timesteps)


class ConvSpikeEncoder(object):
    def __init__(self, conv_weight_np, conv_bias_np, Vthr_in, padding, stride):
        self.scale = 255  # 输入本来就是0-255 不用变， bias和vth需要
        self.Vthr_in = Vthr_in * self.scale

        # todo: conv_bias_np is not used / for snn what is bias?
        # self.conv_bias_np = np.zeros(conv_weight_np.shape[0])
        self.conv_bias_np = conv_bias_np
        self.conv_weight_np = conv_weight_np
        self.conv_weight_np = np.transpose(self.conv_weight_np, (1, 2, 3, 0))

        output_chnl = self.conv_weight_np.shape[3]
        kernel_size = self.conv_weight_np.shape[1]
        input_chnl = self.conv_weight_np.shape[0]

        if padding[0] != padding[1]:
            raise ValueError("padding must be equal on both sides")
        if stride[0] != stride[1]:
            raise ValueError("stride must be equal on both sides")
        padding = padding[0]
        stride = stride[0]
        self.conv_layer = ConvolutionalLayer(
            kernel_size, input_chnl, output_chnl, padding, stride, 1
        )  # k,ic,oc.paddding,stride,speedUP
        self.conv_layer.init_param()
        self.conv_layer.load_param(self.conv_weight_np, self.conv_bias_np * self.scale)

    def generate_spike_mem_potential(self, conv_out):
        out_s = conv_out.shape
        spikes = np.zeros([out_s[0], out_s[1], out_s[2], out_s[3]], dtype=np.int32)
        mem_potential = np.zeros([1, out_s[1], out_s[2], out_s[3]], dtype=np.int32)
        for t in range(out_s[0]):
            mem_potential += conv_out[t]
            spike = mem_potential >= self.Vthr_in
            mem_potential -= spike * self.Vthr_in
            spikes[t] = spike
        return spikes

    def SpikeEncode(self, x, timesteps):
        x = np.expand_dims(x, axis=0)
        conv_out = self.conv_layer.forward(x).astype(np.int32)
        conv_out = conv_out.repeat(timesteps, axis=0)  # SNN spike coding
        spikes = self.generate_spike_mem_potential(conv_out)
        return spikes
