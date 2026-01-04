import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from .utils import is_ci_env, make_dump_dir, measure_time

TEST_TOOLCHAIN_BUILD_DIR = "toolchain_build"


# Add custom markers to eliminate pytest warning
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "perf: mark test to measure performance. Skip if running in ci environment.",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "perf" in item.keywords and is_ci_env():
        pytest.skip("Skipping perf test in CI environment")


@pytest.fixture
def toolchain_build_dir(request: pytest.FixtureRequest) -> Path:
    return Path(request.config.rootpath) / "tests" / TEST_TOOLCHAIN_BUILD_DIR


@pytest.fixture(scope="module")
def ensure_dump_dir(request, tmp_path_factory):
    p = make_dump_dir(request.path.parent, tmp_path_factory)
    yield p


@pytest.fixture(scope="module")
def ensure_dump_dir_and_clean(request, tmp_path_factory):
    p = make_dump_dir(request.path.parent, tmp_path_factory)
    yield p
    for f in p.iterdir():
        f.unlink(missing_ok=True)


@pytest.fixture
def cleandir():
    with tempfile.TemporaryDirectory() as newpath:
        old_cwd = os.getcwd()
        os.chdir(newpath)
        yield
        os.chdir(old_cwd)


@pytest.fixture
def perf_fixture(request):
    with measure_time(f"{request.node.name}"):
        yield


@pytest.fixture(scope="session")
def fixed_rng() -> np.random.Generator:
    return np.random.default_rng(42)
