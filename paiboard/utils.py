from collections.abc import Callable
from typing import TypeVar
import numpy as np
import importlib
import time
import warnings

from contextlib import contextmanager
from functools import wraps
from types import ModuleType

from .exceptions import (
    PAIBoardTimeoutError,
    PAIBoardFileNotFoundError,
    PAIBoardOptionalFileMissingWarning,
)

FT = TypeVar("FT", bound=Callable)


@contextmanager
def wait_with_timeout(timeout: float, err_msg: str):
    start_time = time.perf_counter()
    yield lambda: time.perf_counter() - start_time > timeout

    if time.perf_counter() - start_time > timeout:
        raise PAIBoardTimeoutError(err_msg)


def check_requirements(attr_name: str, fp: str, is_optional: bool = False):
    """Decorate a function that requires an optional attribute to be set."""

    def decorator(func: FT) -> FT:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, attr_name) or not getattr(self, attr_name):
                if is_optional:
                    warnings.warn(
                        f"missing optional file: {fp}",
                        PAIBoardOptionalFileMissingWarning,
                    )
                else:
                    raise PAIBoardFileNotFoundError(f"missing necessary file: {fp}")

            return func(self, *args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def frame_np2txt(frameBuffer, txt_path, frameSplit=True):
    with open(txt_path, "w") as f:
        for i in range(frameBuffer.shape[0]):
            if frameSplit:
                frameStr = "{:064b}".format(frameBuffer[i])
                dataLen = [4, 10, 10, 10, 3, 11, 8, 8]
                for j in range(len(dataLen)):
                    f.write(frameStr[sum(dataLen[:j]) : sum(dataLen[: j + 1])] + " ")
                f.write("\n")
            else:
                f.write("{:064b}\n".format(frameBuffer[i]))


def txtframe2bin(txt_path):
    config_frames = np.loadtxt(txt_path, str)
    config_num = config_frames.size
    config_buffer = np.zeros((config_num,), dtype=np.uint64)
    for i in range(0, config_num):
        config_buffer[i] = int(config_frames[i], 2)
    config_frames = config_buffer
    configPath = txt_path[:-4] + ".bin"
    config_frames.tofile(configPath)


class LazyImport:
    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._module is None:
            try:
                self._module = importlib.import_module(self.module_name)
            except ImportError as e:
                raise ImportError(
                    f"Could not import module '{self.module_name}'"
                ) from e
        return self._module

    def __getattr__(self, name: str):
        module = self._load()
        return getattr(module, name)

    def __repr__(self) -> str:
        if self._module is None:
            return (
                f"<{self.__class__.__name__} pending for module '{self._module_name}'>"
            )
        else:
            return f"<{self.__class__.__name__} for module '{self._module_name}'>"


TIME_UNITS = [("s", 1, 4), ("ms", 1e3, 2), ("us", 1e6, 2)]


def format_elapsed_time(time_s: float) -> str:
    """Format elapsed time in seconds to an appropriate unit.

    Returns:
        Formatted string with appropriate time unit
    """
    for unit, divisor, precision in TIME_UNITS:
        if time_s >= 1.0 / divisor:
            t = time_s * divisor
            return f"{t:.{precision}f} {unit}"

    return f"{time_s:.4f} s"


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


def time_it(func):
    """Decorator to measure execution time of a function. Uses the instance's `_debug_mode` attribute to    \
        determine whether to measure execution time.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the `_debug_mode` attribute of the first argument 'self'
        if args:
            debug_mode = getattr(args[0], "_debug_mode", False)

        if debug_mode:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            time_s = end_time - start_time

            formatted_time = format_elapsed_time(time_s)
            print(f"Function '{func.__name__}' executed in {formatted_time}.")
            return result
        else:
            return func(*args, **kwargs)

    wrapper._real_time_critical = True  # type: ignore
    return wrapper
