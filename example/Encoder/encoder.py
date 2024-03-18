import numpy as np
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
