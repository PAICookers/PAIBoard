import time
from functools import wraps

time_dict = {
    "PoissonEncoder": 0.0,
    "genSpikeFrame ": 0.0,
    "Init          ": 0.0,
    "SendFrame     ": 0.0,
    "RecvFrame     ": 0.0,
    "genOutputSpike": 0.0,
    "FULL INFERENCE": 0.0,
    "CORE INFERENCE": 0.0,
}


# 定义装饰器
def time_calc_addText(fun_name):
    def time_calc(func):
        @wraps(func)
        def wrapper(*args, **kargs):
            t1 = time.time()

            f = func(*args, **kargs)

            t2 = time.time()
            time_dict[fun_name] = time_dict[fun_name] + (t2 - t1) * 1000 * 1000
            return f

        return wrapper

    return time_calc


def get_original_function(decorated_func):
    if hasattr(decorated_func, "__wrapped__"):
        return decorated_func.__wrapped__
    else:
        return decorated_func


def record_time(core_time, full_time):
    time_dict["FULL INFERENCE"] = time_dict["FULL INFERENCE"] + (full_time) * 1000 * 1000
    time_dict["CORE INFERENCE"] = time_dict["CORE INFERENCE"] + int(core_time / 1.0)

def print_time(img_num):
    print("----------------------------------")
    for key in time_dict:
        if (time_dict[key] / img_num) > 1000:
            print(key + " TIME : {:.1f} ms".format(time_dict[key] / img_num / 1000))
        else:
            print(key + " TIME : {:.1f} us".format(time_dict[key] / img_num))
    print("----------------------------------")
