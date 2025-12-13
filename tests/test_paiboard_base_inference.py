import numpy as np
import pytest

from paiboard.base import _multi_inputs_mapping

from .mock_paiboard import TestPAIBoard


def test_compute_chunk_timestep_batch1_reduces_when_lcn_large(paiboard_base):
    # 手工设置一个较大的 lcn，使得需要右移 chunk_ts。
    paiboard_base.inode_attrs_map = {"in0": {"lcn": 1000}}
    ts = paiboard_base._compute_chunk_timestep()
    assert 1 <= ts <= paiboard_base.timestep


def test_compute_chunk_timestep_batch_gt1_uses_timestep(
    toolchain_files, fake_intf, fake_board_cfg
):
    b = TestPAIBoard(
        toolchain_build_dir=toolchain_files,
        timestep=8,
        n_layer=1,
        intf=fake_intf,
        board_cfg=fake_board_cfg,
        batch_size=2,
    )
    # batch_size>1 时，直接返回 timestep。
    assert b._compute_chunk_timestep() == b.timestep


def test_update_batch_ts_and_n_sync_single_batch(paiboard_base):
    paiboard_base.batch_size = 1
    paiboard_base.chunk_ts = paiboard_base.timestep
    imap = {"in0": np.zeros(4)}

    batch_ts, n_sync = paiboard_base._update_batch_ts_and_n_sync(imap, init=True)
    assert batch_ts is None
    assert n_sync == paiboard_base.chunk_ts + paiboard_base.n_layer

    _, n_sync2 = paiboard_base._update_batch_ts_and_n_sync(imap, init=False)
    assert n_sync2 == paiboard_base.chunk_ts


def test_update_batch_ts_and_n_sync_multi_batch(
    toolchain_files, fake_intf, fake_board_cfg
):
    b = TestPAIBoard(
        toolchain_build_dir=toolchain_files,
        timestep=4,
        n_layer=1,
        intf=fake_intf,
        board_cfg=fake_board_cfg,
        batch_size=2,
    )
    # imap 中单一输入，batch 维度为 4。
    imap = {"in0": np.zeros((4, 2))}
    batch_ts, n_sync = b._update_batch_ts_and_n_sync(imap, init=True)
    assert batch_ts == 4
    assert n_sync == batch_ts + b.n_layer


def test_encode_inputs_size_mismatch_raises(paiboard_base):
    # fake PAIRuntime 中，input_rtcfg_map["in0"].size == 4
    paiboard_base.input_rtcfg_map = {
        "in0": type(
            "RT",
            (),
            {"size": 4, "template": np.arange(4, dtype=np.uint64), "is_online": False},
        )()
    }
    imap = {"in0": np.zeros(3)}  # size 不匹配
    with pytest.raises(ValueError):
        paiboard_base._encode_inputs(imap)


def test_encode_inputs_ok(paiboard_base):
    paiboard_base.input_rtcfg_map = {
        "in0": type(
            "RT",
            (),
            {"size": 4, "template": np.arange(4, dtype=np.uint64), "is_online": False},
        )()
    }
    imap = {"in0": np.zeros(4)}
    frames = paiboard_base._encode_inputs(imap)
    assert isinstance(frames, np.ndarray)
    assert frames.ndim == 1


def test_inference_uses_cached_sync_when_unchanged(paiboard_base, fake_intf):
    # 配置 input_rtcfg_map 以便编码成功。
    paiboard_base.input_rtcfg_map = {
        "in0": type(
            "RT",
            (),
            {"size": 4, "template": np.arange(4, dtype=np.uint64), "is_online": False},
        )()
    }
    paiboard_base.onode_attrs_map = {"out0": {}}

    inputs = {"in0": np.zeros(4)}
    # 首次调用，缓存会被建立。
    out1 = paiboard_base.inference(inputs, decoding_output_strict=False)
    # 第二次调用，n_sync 未变化，应复用缓存。
    out2 = paiboard_base.inference(inputs, decoding_output_strict=False)

    # 验证返回类型（单输出时返回 ndarray）
    assert isinstance(out1, np.ndarray)
    assert isinstance(out2, np.ndarray)
    # send_and_recv_frames 至少被调用两次。
    assert len(fake_intf.send_and_recv_calls) >= 2


def test_inference_with_chunk_splits_inputs(toolchain_files, fake_intf, fake_board_cfg):
    b = TestPAIBoard(
        toolchain_build_dir=toolchain_files,
        timestep=4,
        n_layer=1,
        intf=fake_intf,
        board_cfg=fake_board_cfg,
        batch_size=1,
    )
    # 强制 chunk_ts < timestep 以走 chunk 分支。
    b.chunk_ts = 2
    b.inode_attrs_map = {"in0": {"lcn": 1}}
    b.input_rtcfg_map = {
        "in0": type(
            "RT",
            (),
            {"size": 2, "template": np.arange(2, dtype=np.uint64), "is_online": False},
        )()
    }
    b.onode_attrs_map = {"out0": {}}

    inputs = {"in0": np.zeros(4)}
    outputs = b.inference(inputs, decoding_output_strict=False)
    # 验证返回类型（单输出时返回 ndarray）
    assert isinstance(outputs, np.ndarray)
    # 验证 chunk 分支被调用（应该调用多次 send_and_recv_frames）
    assert len(fake_intf.send_and_recv_calls) >= 2  # timestep=4, chunk_ts=2，至少2次


def test_multi_outputs_concat_behaviour(paiboard_base):
    # 单输出时，返回数组本身。
    paiboard_base.onode_attrs_map = {"out0": {}}
    omap_single = {"out0": np.arange(3)}
    val = paiboard_base._multi_outputs_concat(omap_single)
    assert isinstance(val, np.ndarray)

    # 多输出 + 多 chunk 时，先 concat 再返回 dict。
    paiboard_base.onode_attrs_map = {"o1": {}, "o2": {}}
    omap_list = [
        {"o1": np.array([0, 1]), "o2": np.array([2, 3])},
        {"o1": np.array([4, 5]), "o2": np.array([6, 7])},
    ]
    merged = paiboard_base._multi_outputs_concat(omap_list)
    assert set(merged.keys()) == {"o1", "o2"}
    assert merged["o1"].shape[0] == 4


def test_multi_inputs_mapping_array_and_sequence(toolchain_files):
    from paiboard.runtime.types import InputNodeAttrsMap

    inodes_info = InputNodeAttrsMap(
        **{"a": {"lcn": 1}, "b": {"lcn": 1}},
    )

    # 单个 ndarray 只映射到第一个输入节点。
    arr = np.zeros(4)
    imap = _multi_inputs_mapping(arr, inodes_info)
    assert list(imap.keys()) == ["a"]

    # Sequence 映射按顺序 zip。
    seq = [np.zeros(2), np.zeros(2)]
    imap2 = _multi_inputs_mapping(seq, inodes_info)
    assert set(imap2.keys()) == {"a", "b"}
