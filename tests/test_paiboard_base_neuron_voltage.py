from __future__ import annotations

import numpy as np
import pytest

from paiboard.exceptions import PAIBoardFileNotFoundError

from .mock_paiboard import MockCtrlIntf, TestPAIBoard


def test_read_voltage_requires_neu_phy_loc(tmp_path, fake_board_cfg):
    """当缺少 neuron_phy_loc 文件时，经由装饰器应触发文件缺失错误。"""
    workdir = tmp_path / "toolchain"
    workdir.mkdir()

    # 构造最小必需文件集，但不包含 neuron_phy_loc。
    import json
    from paiboard.global_cfg import (
        DEFAULT_FNAME_CORE_PARAMS_CONF,
        DEFAULT_FNAME_INPUT_NODE_INFO,
        DEFAULT_FNAME_OUTPUT_DEST_INFO,
        DEFAULT_FNAME_GRAPH_INFO,
        DEFAULT_FNAME_CONFIG_FILE_WO_SUFFIX,
    )

    (workdir / f"{DEFAULT_FNAME_CONFIG_FILE_WO_SUFFIX}.bin").write_bytes(b"\x00" * 8)
    (workdir / DEFAULT_FNAME_CORE_PARAMS_CONF).write_text(
        json.dumps({"(0,0)": {"(0,0)": {}}})
    )
    (workdir / DEFAULT_FNAME_INPUT_NODE_INFO).write_text(
        json.dumps({"in0": {"lcn": 1, "size": 1}})
    )
    (workdir / DEFAULT_FNAME_OUTPUT_DEST_INFO).write_text(
        json.dumps({"out0": {"(0,0)": {"tick_relative": [0], "addr_axon": [0]}}})
    )
    (workdir / DEFAULT_FNAME_GRAPH_INFO).write_text(json.dumps({"misc": {}}))

    intf = MockCtrlIntf()
    board = TestPAIBoard(
        toolchain_build_dir=workdir,
        timestep=1,
        n_layer=0,
        intf=intf,
        board_cfg=fake_board_cfg,
    )

    # 删除属性以模拟未加载。
    if hasattr(board, "neu_phy_locs_map"):
        delattr(board, "neu_phy_locs_map")

    with pytest.raises(PAIBoardFileNotFoundError):
        board.read_voltage_of_neuron("IF_0")


def test_read_voltage_single_neuron(paiboard_base, fake_intf):
    """指定 neuron 名称时，返回真实 PAIRuntime.decode_voltage 的结果。"""
    # fixture 中已生成 neu_phy_locs_map / neu_vol_reading_frames。
    # 由于 fake_intf.send_and_recv_frames 返回全零数组，decode_voltage 可能返回零值数组
    # 我们只验证返回类型和基本结构，具体数值由 PAIRuntime 的测试保证
    v = paiboard_base.read_voltage_of_neuron("IF_0", is_online=False)
    assert isinstance(v, np.ndarray)
    # 根据 neu_loc 配置，n_neuron=1，所以应该是 (1,) 形状
    assert v.shape == (1,)
    assert v.dtype == np.int32


def test_read_voltage_all_neurons(paiboard_base):
    """不指定 neuron 名称时，应遍历所有神经元节点并聚合结果。"""
    all_v = paiboard_base.read_voltage_of_neuron(is_online=False)
    assert isinstance(all_v, dict)
    # fixture 中只定义了一个节点 IF_0。
    assert set(all_v.keys()) == {"IF_0"}
    # 根据 neu_loc 配置，n_neuron=1
    assert all_v["IF_0"].shape == (1,)
    assert all_v["IF_0"].dtype == np.int32


def test_read_voltage_invalid_neuron_name_raises(paiboard_base):
    """非法 neuron 名字符串应触发 ValueError。"""
    with pytest.raises(ValueError):
        paiboard_base.read_voltage_of_neuron("NON_EXIST")
