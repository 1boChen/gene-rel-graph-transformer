# tests/test_labels.py

from gene_rel_gt.preprocessing.labels import subtype_to_vector


def test_subtype_to_vector_single_label():
    vec = subtype_to_vector("activation")
    assert sum(vec) == 1


def test_subtype_to_vector_multi_label():
    vec = subtype_to_vector("activation,, inhibition")
    assert sum(vec) == 2


def test_subtype_to_vector_no_interaction():
    vec = subtype_to_vector("no_interaction")
    assert sum(vec) == 0
