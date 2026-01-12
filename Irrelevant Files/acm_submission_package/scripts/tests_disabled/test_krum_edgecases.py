import numpy as np
import fl_helpers


def test_krum_single_client():
    # Single client: should return the client's update
    client = [np.array([0.1, 0.2])]
    out = fl_helpers.federated_krum([client])
    # for per-layer list input, federated_krum returns the selected client's list
    assert isinstance(out, list)
    assert len(out) == len(client)
    assert np.allclose(out[0], client[0])


def test_krum_two_clients_same_shape():
    a = [np.array([0.1, 0.2])]
    b = [np.array([0.2, 0.1])]
    out = fl_helpers.federated_krum([a, b])
    assert isinstance(out, list)
    assert len(out) == 1
    # output must equal one of the inputs
    eq_a = all(np.allclose(o, ia) for o, ia in zip(out, a))
    eq_b = all(np.allclose(o, ib) for o, ib in zip(out, b))
    assert eq_a or eq_b


def test_krum_mixed_layer_shapes():
    # Mixed per-layer shapes across clients
    c1 = [np.array([1.0]), np.array([1.0, 2.0])]
    c2 = [np.array([2.0]), np.array([2.0, 4.0])]
    c3 = [np.array([10.0]), np.array([10.0, 20.0])]
    out = fl_helpers.federated_krum([c1, c2, c3])
    # Should return one of the client lists and not raise
    assert isinstance(out, list)
    assert len(out) == len(c1)
