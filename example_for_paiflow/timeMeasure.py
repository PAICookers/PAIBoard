import time
from functools import wraps

time_dict = {'runPreNet     ': 0.0, 'Tensor2Frame  ': 0.0, 'Init          ': 0.0, 'SendFrame     ': 0.0, 'RecvFrame     ': 0.0, 'Frame2Tensor  ': 0.0, 'FULL INFERENCE': 0.0, 'CORE INFERENCE': 0.0}

# 定义装饰器
def time_calc_addText(fun_name):
    def time_calc(func):
        @wraps(func)
        def wrapper(*args, **kargs):        
            t1 = time.time()

            f = func(*args,**kargs)     

            t2 = time.time()
            time_dict[fun_name] = time_dict[fun_name] + (t2-t1)*1000*1000
            return f    
        return wrapper
    return time_calc

def get_original_function(decorated_func):
    if hasattr(decorated_func, '__wrapped__'):
        return decorated_func.__wrapped__
    else:
        return decorated_func
