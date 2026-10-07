import time
from datetime import datetime

import pytest

from exercises.week01 import ex01_registries_and_kwargs as ex01
from exercises.week01 import ex02_closures as ex02
from exercises.week01 import ex03_abc_and_naming as ex03
from exercises.week01 import ex04_context_and_decorators as ex04

# ----------------------------------------------------------------------------- ex01


def test_build_norm_known_names():
    assert ex01.build_norm("LayerNorm", 16) == ex01.LayerNorm(16)
    assert ex01.build_norm("RMSNorm", 8, eps=1e-3) == ex01.RMSNorm(8, eps=1e-3)
    assert ex01.build_norm(None, 8) is None


def test_build_norm_unknown_name_lists_options():
    with pytest.raises(ValueError, match="LayerNorm") as err:
        ex01.build_norm("LayerNrom", 16)
    for name in ex01.NORM_TYPES:
        assert name in str(err.value)


@pytest.mark.parametrize("attn_type", ["torch", "flex", "flash"])
def test_layer_configs_basic(attn_type):
    configs = ex01.build_layer_configs(3, 64, attn_type=attn_type, window_size=128, norm="RMSNorm", attn_kwargs={"num_heads": 4})
    assert [c.depth for c in configs] == [0, 1, 2]
    assert all(c.norm == "RMSNorm" for c in configs)
    assert all(c.attn.attn_type == attn_type for c in configs)
    assert all(c.attn.num_heads == 4 for c in configs)
    assert [c.attn.is_first_layer for c in configs] == [True, False, False]
    expected_window = 128 if attn_type == "flash" else None
    assert all(c.attn.window_size == expected_window for c in configs)


def test_layer_configs_are_independent():
    shared = {"hidden": 32}
    configs = ex01.build_layer_configs(2, 16, dense_kwargs=shared, attn_kwargs=None)
    configs[0].dense_kwargs["hidden"] = 999
    assert configs[1].dense_kwargs["hidden"] == 32
    assert shared["hidden"] == 32, "the caller's dict must not be modified"


def test_layer_configs_do_not_mutate_caller_attn_kwargs():
    attn_kwargs = {"bias": False}
    ex01.build_layer_configs(2, 16, attn_type="flash", window_size=8, attn_kwargs=attn_kwargs)
    assert attn_kwargs == {"bias": False}


def test_collect_names_fresh_default():
    assert ex01.collect_names("a") == ["a"]
    assert ex01.collect_names("b") == ["b"], "the default list must not be shared between calls"
    existing = ["x"]
    assert ex01.collect_names("y", existing) is existing
    assert existing == ["x", "y"]


# ----------------------------------------------------------------------------- ex02


def test_window_mask():
    mask = ex02.make_window_mask(4)
    assert mask(5, 5)
    assert mask(5, 7)
    assert mask(7, 5)
    assert not mask(5, 8)


def test_window_masks_are_independent():
    small, big = ex02.make_window_mask(2), ex02.make_window_mask(10)
    assert not small(0, 3)
    assert big(0, 3)


def test_wrapped_window_reads_live_length():
    n = [10]
    mask = ex02.make_wrapped_window_mask(4, n)
    assert mask(0, 9)
    assert mask(1, 9)
    assert not mask(0, 5)
    n[0] = 100
    assert not mask(0, 9), "the closure must read seq_len[0] when called, not when created"


def test_counter():
    a, b = ex02.make_counter(), ex02.make_counter()
    assert [a(), a(), a()] == [1, 2, 3]
    assert b() == 1


def test_multipliers_late_binding():
    fs = ex02.make_multipliers(4)
    assert [f(10) for f in fs] == [0, 10, 20, 30]


# ----------------------------------------------------------------------------- ex03


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("hit_on_valid_particle", ("hit", "on_valid_particle")),
        ("pix_x", ("pix", "x")),
        ("strip_embed", ("strip", "embed")),
    ],
)
def test_split_key(key, expected):
    assert ex03.split_key(key, ["hit", "pix", "strip"]) == expected


def test_split_key_needs_underscore():
    with pytest.raises(KeyError):
        ex03.split_key("hitx_y", ["hit"])


def _two_inputs():
    return {"pix_embed": [1, 2], "pix_valid": [True, True], "strip_embed": [3], "strip_valid": [False]}


def test_merge_inputs():
    x = _two_inputs()
    merged = ex03.merge_inputs(x, ["pix", "strip"])
    assert merged["key_embed"] == [1, 2, 3]
    assert merged["key_valid"] == [True, True, False]
    assert merged["key_is_pix"] == [True, True, False]
    assert merged["key_is_strip"] == [False, False, True]
    assert "key_embed" not in x, "merge_inputs must return a new dict"


def test_unmerge_roundtrip():
    merged = ex03.merge_inputs(_two_inputs(), ["pix", "strip"])
    merged["key_embed"] = [10, 20, 30]  # pretend the encoder updated the embeddings
    out = ex03.unmerge_inputs(merged, ["pix", "strip"])
    assert out["pix_embed"] == [10, 20]
    assert out["strip_embed"] == [30]


def test_task_is_abstract():
    with pytest.raises(TypeError):
        ex03.Task("t", "hit")


def test_threshold_task():
    task = ex03.ThresholdTask("filter", "hit", weight=2.0, threshold=1.0)
    assert task.name == "filter"
    assert task.input_object == "hit"
    out = task.forward({"hit_embed": [0.1, 0.5, 2.0]})
    assert out == {"hit_score": [0.2, 1.0, 4.0]}
    assert task.predict(out) == {"hit_pred": [False, True, True]}
    assert task.attn_mask(out) == {}


# ----------------------------------------------------------------------------- ex04


def test_timer_class():
    t = ex04.Timer()
    assert t.elapsed is None
    with t as same:
        time.sleep(0.01)
    assert same is t
    assert t.elapsed >= 0.009


def test_timer_class_records_on_exception():
    t = ex04.Timer()
    with pytest.raises(ValueError, match="boom"), t:
        raise ValueError("boom")
    assert t.elapsed is not None


def test_timer_generator():
    with ex04.timer() as result:
        time.sleep(0.01)
    assert result["elapsed"] >= 0.009
    with pytest.raises(KeyError), ex04.timer() as result2:
        raise KeyError("x")
    assert "elapsed" in result2


def test_log_calls():
    @ex04.log_calls
    def add(a, b=0):
        """Add two numbers."""
        return a + b

    assert add(1, b=2) == 3
    assert add(5) == 5
    assert add.__name__ == "add"
    assert add.__doc__ == "Add two numbers."
    assert add.calls == [((1,), {"b": 2}), ((5,), {})]


def test_rundir_roundtrip():
    run = ex04.RunDir("my_run", datetime(2026, 10, 2, 12, 31, 10))
    assert run.dirname == "my_run_20261002-T123110"
    back = ex04.RunDir.from_dirname(run.dirname)
    assert back.name == "my_run"
    assert back.timestamp == datetime(2026, 10, 2, 12, 31, 10)
    with pytest.raises(ValueError):
        ex04.RunDir.from_dirname("my_run_yesterday")
