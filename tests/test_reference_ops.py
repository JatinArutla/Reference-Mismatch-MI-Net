import numpy as np

from refshift.preprocessing.reference_ops import car, median_ref, gram_schmidt_ref, apply_reference


def test_car_zero_common_mode():
    x = np.array([[1., 2., 3.], [4., 5., 6.]])
    y = car(x)
    assert np.allclose(y.mean(axis=0), 0.0)


def test_median_shape():
    x = np.random.randn(5, 100)
    y = median_ref(x)
    assert y.shape == x.shape


def test_gs_shape():
    x = np.random.randn(6, 200)
    y = gram_schmidt_ref(x)
    assert y.shape == x.shape


def test_apply_native():
    x = np.random.randn(4, 50)
    y = apply_reference(x, "native")
    assert np.allclose(x, y)
