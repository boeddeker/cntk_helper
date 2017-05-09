import numpy as np

import cntk


def assert_allclose(
        x,
        y,
        *,
        rtol=1e-7,
        atol=1e-7,
        xlabel='x',
        ylabel='y'
):
    if not isinstance(x, dict) and not isinstance(y, dict):
        x = {'input': x}
        y = {'input': y}
    elif not isinstance(x, dict):
        assert len(y) == 1, len(y)
        x = {list(y.keys())[0]: x}
    elif not isinstance(y, dict):
        assert len(x) == 1, len(x)
        y = {list(x.keys())[0]: y}
    else:
        assert len(x) == len(y)
        assert set(x.keys()) == set(y.keys()), (set(x.keys()), set(y.keys()))

    for k in x.keys():
        try:
            np.testing.assert_allclose(x[k], y[k], rtol=rtol, atol=atol)
        except AssertionError as e:
            x = x[k]
            y = y[k]
            e.args = [(
                f'Key: {k}\n'
                f'Minimum atol when rtol is zero: {np.abs(x - y).max()}\n'
                f'Minimum rtol when atol is zero: '
                f'{(np.abs(x - y)/abs(y)).max()}'
                '\n'
                f'Minimum atol when rtol is {rtol}: '
                f'{(np.abs(x - y) - rtol * abs(y)).max()}\n'
                f'Minimum rtol when atol is {atol}: '
                f'{((np.abs(x - y) - atol)/abs(y)).max()}\n\n'
                f'x: {xlabel}\n'
                f'y: {ylabel}\n'
                f'\n{e.args[0] if len(e.args) == 1 else e.args}'
            )]
            raise


def get_full_shape(cntk_variable, numpy_array):
    """
    cntk_variable:
    numpy_array: numpy array, that has the same dynamic axis as cntk_variable
    """
    shape = cntk_variable.shape
    shape = numpy_array.shape[:len(cntk_variable.dynamic_axes)] + shape
    return shape


def test_numerical_grad(
        y,
        arguments,
        grad_outputs,
        eps=1e-3,
        rtol=1e-7,
        atol=1e-7,
):
    """
    Not Supported:
        - Different sequence length
          (everything has to be a ndarray and not list of ndarray)
        - Two or more outputs
    """
    assert eps > 0
    assert isinstance(arguments, dict)

    # if grad_outputs is None:
    #     # Try to guess shape
    #     assert len(arguments) == 1, 'Could not guess grad_outputs shape'
    #     grad_outputs_shape = get_full_shape(y, list(arguments.values())[0])
    #     grad_outputs = np.random.normal(size=grad_outputs_shape)
    grad_outputs = np.array(grad_outputs, dtype=y.dtype)

    for k, v in arguments.items():
        assert k.needs_gradient is True, \
            (k, 'needs_gradient has to be True')

    tmp = cntk.input(y.shape, dynamic_axes=y.dynamic_axes, dtype=y.dtype)

    grads = {k: np.zeros_like(v) for k, v in arguments.items()}

    cntk_grads = (y * tmp).grad({tmp: np.array(grad_outputs, dtype=y.dtype),
                                 **arguments})
    if len(arguments) == 1 and not isinstance(cntk_grads, dict):
        cntk_grads = {list(arguments.keys())[0]: cntk_grads}

    for k in arguments.keys():
        for i in np.ndindex(arguments[k].shape):
            orig = arguments[k][i]
            arguments[k][i] = orig + eps
            ys1 = np.array(y.eval(arguments))
            arguments[k][i] = orig - eps
            ys2 = np.array(y.eval(arguments))
            arguments[k][i] = orig

            grads[k][i] += np.sum((ys1 - ys2) * grad_outputs) / (2 * eps)

        cntk_grads[k] = np.array(cntk_grads[k])
        if grads[k].ndim == cntk_grads[k].ndim - 1 \
                and cntk_grads[k].shape[0] == 1:
            grads[k] = grads[k][None, ...]

    assert_allclose(cntk_grads, grads, rtol=rtol, atol=atol)
    return grads


if __name__ == '__main__':

    x = cntk.input(3, dtype=np.float64, needs_gradient=True)

    y = x + x

    x0 = np.random.normal(size=(3))
    arguments = {x: x0}
    grad_outputs = [1, 2, 3]

    print(test_numerical_grad(y, arguments, grad_outputs))

    x = cntk.sequence.input(3, dtype=np.float64, needs_gradient=True)

    y = x + x

    x0 = np.random.normal(size=(1, 2, 3))
    arguments = {x: x0}
    grad_outputs = [[[1, 2, 3], [4, 5, 6]]]

    print(test_numerical_grad(y, arguments, grad_outputs))
