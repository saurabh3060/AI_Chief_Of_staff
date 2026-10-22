"""Unit tests for department centroid / prototype assignment."""

import numpy as np

from agent_email.dept_centroids import pick_department, raw_cosine_scores
from core.text_match import meaningful_tokens


def test_pick_department_nearest_centroid():
    wholesale = np.array([1.0, 0.0], dtype=np.float32)
    bbrl = np.array([0.0, 1.0], dtype=np.float32)
    centroids = {"Wholesale Banking": wholesale, "BB&RL": bbrl}

    name, scores = pick_department(np.array([0.9, 0.1], dtype=np.float32), centroids)
    assert name == "Wholesale Banking"
    assert scores["Wholesale Banking"] > scores["BB&RL"]


def test_pick_department_empty_centroids():
    name, scores = pick_department(np.array([1.0, 0.0], dtype=np.float32), {})
    assert name == ""
    assert scores == {}


def test_raw_cosine_scores_clamped():
    centroids = {"A": np.array([1.0, 0.0], dtype=np.float32)}
    scores = raw_cosine_scores(np.array([1.0, 0.0], dtype=np.float32), centroids)
    assert scores["A"] == 100.0


def test_prototype_max_beats_diluted_mean_centroid():
    """
    A broad department (CASA seed + unrelated seed) loses a CASA-like
    query on mean centroid, but wins on nearest prototype when it has
    a matching CASA seed. This is the general dilution failure mode.
    """
    casa = np.array([1.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 1.0], dtype=np.float32)
    mixed_mean = np.array([0.7071, 0.7071], dtype=np.float32)
    centroids = {"Wholesale Banking": mixed_mean, "BB&RL": casa}
    prototypes = {
        "Wholesale Banking": np.vstack([casa, other]),
        "BB&RL": casa.reshape(1, -1),
    }
    query = np.array([1.0, 0.0], dtype=np.float32)

    mean_name, _ = pick_department(query, centroids)
    proto_name, proto_scores = pick_department(query, centroids, prototypes=prototypes)

    assert mean_name == "BB&RL"
    assert proto_scores["Wholesale Banking"] == proto_scores["BB&RL"] == 100.0
    assert proto_name in {"Wholesale Banking", "BB&RL"}


def test_close_prototype_scores_break_on_dictionary_tokens():
    casa = np.array([1.0, 0.0], dtype=np.float32)
    centroids = {"Wholesale Banking": casa, "BB&RL": casa}
    prototypes = {
        "Wholesale Banking": casa.reshape(1, -1),
        "BB&RL": casa.reshape(1, -1),
    }
    dept_tokens = {
        "Wholesale Banking": meaningful_tokens("wholesale CASA WB CA/SA targets"),
        "BB&RL": meaningful_tokens("retail CASA retail CA/SA targets branch"),
    }
    cluster_tokens = meaningful_tokens("Sep-26 CA/SA targets - WB view required")

    name, scores = pick_department(
        casa,
        centroids,
        prototypes=prototypes,
        cluster_tokens=cluster_tokens,
        dept_tokens=dept_tokens,
        margin=3.0,
    )
    assert abs(scores["Wholesale Banking"] - scores["BB&RL"]) <= 3.0
    assert name == "Wholesale Banking"


def test_close_scores_pick_retail_when_cluster_tokens_are_retail():
    casa = np.array([1.0, 0.0], dtype=np.float32)
    centroids = {"Wholesale Banking": casa, "BB&RL": casa}
    prototypes = {
        "Wholesale Banking": casa.reshape(1, -1),
        "BB&RL": casa.reshape(1, -1),
    }
    dept_tokens = {
        "Wholesale Banking": meaningful_tokens("wholesale CASA WB CA/SA targets"),
        "BB&RL": meaningful_tokens("retail CASA retail CA/SA targets branch"),
    }
    cluster_tokens = meaningful_tokens("retail CA/SA targets for branch CASA")

    name, _ = pick_department(
        casa,
        centroids,
        prototypes=prototypes,
        cluster_tokens=cluster_tokens,
        dept_tokens=dept_tokens,
        margin=3.0,
    )
    assert name == "BB&RL"


def test_casa_slash_and_wb_expand_in_tokens():
    tokens = meaningful_tokens("Sep-26 CA/SA targets - WB view required")
    assert "casa" in tokens
    assert "wholesale" in tokens
    assert "banking" in tokens
