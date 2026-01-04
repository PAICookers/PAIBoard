import shutil
import warnings
from contextlib import nullcontext

import numpy as np
import pytest

from paiboard.base import PAIBoard
from paiboard.exceptions import (
    PAIBoardFileNotFoundError,
    PAIBoardOptionalFileMissingWarning,
)
from paiboard.global_cfg import (
    DEFAULT_FNAME_CONFIG_FILE_WO_SUFFIX,
    DEFAULT_FNAME_CORE_PARAMS_CONF,
    DEFAULT_FNAME_INPUT_NODE_INFO,
    DEFAULT_FNAME_OUTPUT_DEST_INFO,
)

from .mock_paiboard import MockPAIBoard
from .utils import dir_not_exist_fail, gen_random_array


class TestPAIBoard:
    def test_init_workdir_not_dir(self):
        with pytest.raises(ValueError):
            _ = PAIBoard(toolchain_build_dir="not_exist_dir", timestep=8, n_layer=4)

    @pytest.mark.parametrize(
        "missing_name, exc_type",
        [
            (DEFAULT_FNAME_CORE_PARAMS_CONF, PAIBoardFileNotFoundError),
            (DEFAULT_FNAME_INPUT_NODE_INFO, PAIBoardFileNotFoundError),
            (DEFAULT_FNAME_OUTPUT_DEST_INFO, PAIBoardFileNotFoundError),
            (f"{DEFAULT_FNAME_CONFIG_FILE_WO_SUFFIX}.bin", PAIBoardFileNotFoundError),
        ],
    )
    def test_init_missing_required_files_raises(
        self, tmp_path, toolchain_build_dir, missing_name, exc_type
    ):
        model_dir = toolchain_build_dir / "test_001_Conv1d"
        dir_not_exist_fail(model_dir)
        shutil.copytree(model_dir, tmp_path, dirs_exist_ok=True)

        target = tmp_path / missing_name
        target.unlink()

        with pytest.raises(exc_type):
            _ = MockPAIBoard(tmp_path, 8, 4)

    def test_init_optional_files_missing_only_warn(self, tmp_path, toolchain_build_dir):
        model_dir = toolchain_build_dir / "test_001_Conv1d"
        dir_not_exist_fail(model_dir)
        shutil.copytree(model_dir, tmp_path, dirs_exist_ok=True)

        with pytest.warns(PAIBoardOptionalFileMissingWarning):
            _ = MockPAIBoard(tmp_path, 4, 1)

    @pytest.mark.parametrize(
        "data, timestep, batch_mode, expectation",
        [
            (np.ones((3, 28, 28)), 3, False, pytest.warns(UserWarning)),
            (np.ones((1, 8, 3 * 32 * 32)), 8, True, nullcontext()),
            (np.ones((8, 32, 32)), 8, False, pytest.warns(UserWarning)),
            (np.ones((4, 1, 64, 64)), 1, True, pytest.warns(UserWarning)),
            (np.ones((1, 16, 32, 32)), 16, True, pytest.warns(UserWarning)),
            (np.ones((1, 8, 32 * 32)), 8, False, pytest.raises(ValueError)),
            (np.ones((4, 8, 32 * 32)), 4, True, pytest.raises(ValueError)),
            (np.ones((8, 32, 32, 32)), 8, True, pytest.raises(ValueError)),
            (np.ones((8, 32, 32, 32)), 8, False, nullcontext()),
        ],
    )
    def test_reshape_inputs(self, data, timestep, batch_mode, expectation):
        def _validate_and_reshape_inputs(data: np.ndarray):
            if not batch_mode:
                if data.ndim < 2:
                    raise ValueError()
                elif data.shape[0] != timestep:
                    raise ValueError()

                reshaped = data.reshape(1, timestep, -1)
                if data.ndim > 2:
                    warnings.warn("reshaped")

            else:
                if data.ndim < 3:
                    raise ValueError()
                elif data.shape[1] != timestep:
                    raise ValueError()

                reshaped = data.reshape(data.shape[0], timestep, -1)
                if data.ndim > 3:
                    warnings.warn("reshaped")

            return reshaped

        with expectation:
            _ = _validate_and_reshape_inputs(data)

    def test_inference_no_batch_no_chunk(self, toolchain_build_dir):
        model_dir = toolchain_build_dir / "test_001_Conv1d"
        dir_not_exist_fail(model_dir)

        # input node LCN=1, ts=8, layer=1, input size (800,)
        ishape = (800,)
        timestep = 8
        n_layer = 1
        pb = MockPAIBoard(model_dir, timestep, n_layer, batch_mode=False)

        if pb._chunk_ts < timestep:
            pytest.fail("timestep need be smaller")

        # Prepare input, ts*size
        ishape_with_ts = (timestep, *ishape)
        oshape_with_ts = (timestep, -1)

        data = gen_random_array(ishape_with_ts, dtype=np.uint8)
        out = pb.inference(data)

        assert out.ndim == 2  # Not in batch mode
        assert oshape_with_ts[:-1] == out.shape[:-1]

        # Raise exception if input shape has batch size dimension
        with pytest.raises(ValueError):
            ishape_with_ts = (1, timestep, *ishape)
            data = gen_random_array(ishape_with_ts, dtype=np.uint8)
            _ = pb.inference(data)

    @pytest.mark.parametrize("timestep", [128, 200])
    def test_inference_no_batch_with_chunk(self, toolchain_build_dir, timestep):
        model_dir = toolchain_build_dir / "test_005_Conv2dSemiFolded"
        dir_not_exist_fail(model_dir)

        # input node LCN=4X, ts=128, layer=3, input size (64,8)
        ishape = (64, 8)
        n_layer = 3
        pb = MockPAIBoard(model_dir, timestep, n_layer, batch_mode=False)

        if pb._chunk_ts >= timestep:
            pytest.fail("timestep need be larger")

        # Prepare input
        ishape_with_ts = (timestep, *ishape)
        oshape_with_ts = (timestep, -1)

        data = gen_random_array(ishape_with_ts, dtype=np.uint8)
        out = pb.inference(data)

        assert out.ndim == 2  # Not in batch mode
        assert oshape_with_ts[:-1] == out.shape[:-1]

        # Raise exception if input shape is (batch size, timestep, ...)
        with pytest.raises(ValueError):
            ishape_with_ts = (1, timestep, *ishape)
            data = gen_random_array(ishape_with_ts, dtype=np.uint8)
            _ = pb.inference(data)

    @pytest.mark.parametrize("total_batch", [1, 4, 20])
    def test_inference_with_batch_no_chunk(self, toolchain_build_dir, total_batch):
        model_dir = toolchain_build_dir / "test_005_Conv2dSemiFolded"
        dir_not_exist_fail(model_dir)

        # input node LCN=4X, ts=8, layer=3, input size (64,8)
        ishape = (64, 8)
        timestep = 8
        n_layer = 3
        pb = MockPAIBoard(model_dir, timestep, n_layer, batch_mode=True)

        if pb._chunk_ts < timestep:
            pytest.fail("timestep need be smaller")

        # Prepare input
        ishape_with_bs_ts = (total_batch, timestep, *ishape)
        oshape_with_bs_ts = (total_batch, timestep, -1)

        data = gen_random_array(ishape_with_bs_ts, dtype=np.uint8)
        out = pb.inference(data)

        assert oshape_with_bs_ts[:-1] == out.shape[:-1]

        # Raise exception if input shape is (timestep, ...)
        with pytest.raises(ValueError):
            ishape_with_ts = (timestep, *ishape)
            data = gen_random_array(ishape_with_ts, dtype=np.uint8)
            _ = pb.inference(data)

    @pytest.mark.parametrize("timestep", [128, 200])
    @pytest.mark.parametrize("total_batch", [1, 4, 8])
    def test_inference_with_batch_with_chunk(
        self, toolchain_build_dir, timestep, total_batch
    ):
        model_dir = toolchain_build_dir / "test_005_Conv2dSemiFolded"
        dir_not_exist_fail(model_dir)

        # input node LCN=4X, ts=8, layer=3, input size (64,8)
        ishape = (64, 8)
        n_layer = 3
        pb = MockPAIBoard(model_dir, timestep, n_layer, batch_mode=True)

        if pb._chunk_ts >= timestep:
            pytest.fail("timestep need be larger")

        # Prepare input
        ishape_with_bs_ts = (total_batch, timestep, *ishape)
        oshape_with_bs_ts = (total_batch, timestep, -1)

        data = gen_random_array(ishape_with_bs_ts, dtype=np.uint8)
        out = pb.inference(data)

        # Check the size of batch_size & timestep
        assert oshape_with_bs_ts[:-1] == out.shape[:-1]


class TestPAIBoardZynq:
    pass
