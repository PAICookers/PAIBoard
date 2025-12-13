class PAIBoardException(Exception):
    """Base class for exceptions in this module."""

    pass


class PAIBoardWarning(UserWarning):
    """Base class for warnings in this module."""

    pass


class BoardCfgError(PAIBoardException):
    """Exception for incorrect board configuration."""

    pass


class PAIBoardFileNotFoundError(PAIBoardException, FileNotFoundError):
    """Exception for missing necessary files."""

    pass


class PAIBoardPlatformNotSupportedWarning(PAIBoardWarning, RuntimeWarning):
    """Warning for certain features not supported by the current platform."""

    pass


class PAIBoardOptionalFileMissingWarning(PAIBoardWarning, ResourceWarning):
    """Warning for missing optional files."""

    pass


class PAIBoardError(PAIBoardException, ResourceWarning):
    """Exception for missing optional files."""

    pass


class PAIBoardRuntimeError(PAIBoardException, RuntimeError):
    """Exception for runtime error."""

    pass


class PAIBoardRuntimeWarning(PAIBoardWarning, RuntimeWarning):
    """Exception for runtime warning."""

    pass


class PAIBoardDeviceError(PAIBoardRuntimeError, IOError):
    """Basic exception for device error."""

    pass


class PAIBoardDMADeviceError(PAIBoardDeviceError):
    """Exception for DMA device error."""

    pass


class PAIBoardTimeoutError(PAIBoardRuntimeError):
    """Exception for timeout error."""

    pass
