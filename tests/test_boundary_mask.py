"""Unit tests for the generalized boundary-escape position mask.

Run:  python3 -m pytest tests/test_boundary_mask.py -q
"""

import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from span_crc import (boundary_escape_unit, build_spans, load_cache,  # noqa: E402
                      calibration_units, crc_lambda, position_mask, _by_doc)

RESULTS = os.path.join(REPO, "results", "local_20260803")
CACHE = os.path.join(REPO, "results", "cache")


# ------------------------------------------------------------------ masks

def test_mask_shapes():
    for L in (1, 2, 3, 5, 20):
        e = position_mask("edges", L)
        assert e[0] and e[-1] and e.sum() == (1 if L == 1 else 2)
        f = position_mask("first", L)
        assert f[0] and f.sum() == 1
        g = position_mask("last", L)
        assert g[-1] and g.sum() == 1
        i = position_mask("interior", L)
        assert i.sum() == max(0, L - 2)
        assert not i[0] and not i[-1]


def test_mask_unknown_raises():
    with pytest.raises(ValueError):
        position_mask("post_hoc_favourite", 5)


# ------------------------------------- default mask reproduces the paper

@pytest.mark.parametrize("dataset", ["sroie", "funsd"])
def test_default_reproduces_committed_artifact(dataset):
    """The mask refactor must be bit-identical to the committed edge test."""
    ref_path = os.path.join(RESULTS, f"span_crc_{dataset}.json")
    if not os.path.exists(ref_path):
        pytest.skip("committed artifact missing")
    ref = json.load(open(ref_path))["per_unit"][0]["boundary_escape"]

    z, meta, id2label = load_cache(dataset, CACHE)
    name, cal_key, _ = calibration_units(meta)[0]
    cal_spans, cal_lac = build_spans(z[f"{cal_key}_probs"],
                                     z[f"{cal_key}_labels"],
                                     z[f"{cal_key}_docstart"], id2label)
    lam0, _ = crc_lambda(_by_doc(cal_spans), 0.10)
    got = boundary_escape_unit(cal_spans, cal_lac, z[f"{cal_key}_probs"],
                               id2label, tau=lam0, n_boot=ref["n_boot"],
                               seed=0, gamma=ref["gamma"], mask="edges")
    assert got["rejected_fields"] == ref["rejected_fields"]
    for f, rf in ref["fields"].items():
        gf = got["fields"][f]
        assert gf["edge_observed"] == rf["edge_observed"]
        assert gf["n_informative_spans"] == rf["n_informative_spans"]
        if rf["p_value"] is None:
            assert gf["p_value"] is None
        else:
            assert gf["p_value"] == pytest.approx(rf["p_value"], abs=0)
        if rf["edge_expected"]:
            assert gf["edge_expected"] == pytest.approx(rf["edge_expected"])


# ------------------------------------------------- synthetic mask behavior

def _synthetic(n_docs=40, spans_per_doc=4, L=6, edge_hot=True, seed=7):
    """Token stream whose escapes concentrate on edges (or uniformly)."""
    rng = np.random.default_rng(seed)
    id2label = {0: "O", 1: "B-F", 2: "I-F"}
    probs, labels, docstart = [], [], [0]
    for _ in range(n_docs):
        for _ in range(spans_per_doc):
            tags = [1] + [2] * (L - 1)
            for pos, t in enumerate(tags):
                if edge_hot:
                    hot = pos in (0, L - 1) and rng.random() < 0.5
                else:
                    hot = rng.random() < (1.0 / L)
                p = np.full(3, 0.5)
                p[t] = 0.1 if hot else 99.0     # gold prob ~0.09 hot, ~0.99 cold
                probs.append(p / p.sum())       # escape iff 1-p_gold > tau=0.5
                labels.append(t)
        docstart.append(len(labels))
    return (np.array(probs), np.array(labels), np.array(docstart), id2label)


def test_edge_signal_found_by_edges_not_interior():
    probs, labels, docstart, id2label = _synthetic(edge_hot=True)
    spans, lac = build_spans(probs, labels, docstart, id2label)
    kw = dict(cal_probs=probs, id2label=id2label, tau=0.5,
              n_boot=2000, seed=0, gamma=0.10)
    edges = boundary_escape_unit(spans, lac, mask="edges", **kw)
    interior = boundary_escape_unit(spans, lac, mask="interior", **kw)
    assert edges["rejected_fields"] == ["F"]
    # escapes concentrated on edges => interior count is *depleted*, so the
    # one-sided (>=) interior test must not reject.
    assert interior["rejected_fields"] == []


def test_uniform_null_not_rejected():
    probs, labels, docstart, id2label = _synthetic(edge_hot=False)
    spans, lac = build_spans(probs, labels, docstart, id2label)
    for mask in ("edges", "interior", "first", "last"):
        out = boundary_escape_unit(spans, lac, probs, id2label, tau=0.5,
                                   n_boot=2000, seed=0, gamma=0.10, mask=mask)
        assert out["rejected_fields"] == [], mask


def test_interior_degenerate_on_short_spans():
    """Interior mask on spans of length <= 2 tests nothing (blind spot)."""
    probs, labels, docstart, id2label = _synthetic(L=2, edge_hot=True)
    spans, lac = build_spans(probs, labels, docstart, id2label)
    out = boundary_escape_unit(spans, lac, probs, id2label, tau=0.5,
                               n_boot=500, seed=0, gamma=0.10, mask="interior")
    assert out["n_fields_tested"] == 0
