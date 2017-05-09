import numpy as np

import cntk
from cntk import output_variable, user_function
from cntk.ops.functions import UserFunction

from cntk_helper.numerical_grad import test_numerical_grad


def sequence_normalisation(
        init_mean=0,
        init_std=1,
        name=''
):

    beta = cntk.parameter(shape=tuple(), init=init_mean, name='beta')
    gamma = cntk.parameter(shape=tuple(), init=init_std, name='gamma')

    @cntk.layers.blocks.BlockFunction('SequenceNormalisation', name)
    def sequence_normalisation(x):

        # ToDo: Replace this dirty hack with the len of the sequence axis
        sequence_length = cntk.sequence.reduce_sum(cntk.reduce_sum(x) * 0 + 1)

        mean = cntk.sequence.reduce_sum(x) / sequence_length
        x = x - cntk.sequence.broadcast_as(mean, x)
        var = cntk.sequence.reduce_sum(cntk.ops.pow(x, 2)) / sequence_length
        x = x / cntk.sequence.broadcast_as(cntk.sqrt(var), x)

        return x * gamma + beta
    return sequence_normalisation


class MySequenceNormalisation(UserFunction):

    def __init__(self, arg1, axis=0, name='f1'):
        super().__init__([arg1], name=name)
        self.axis = axis

    def forward(self, arguments, device=None, outputs_to_retain=None):
        mean = np.mean(arguments[0], axis=self.axis, keepdims=True)
        mean_free = arguments[0] - mean
        std = np.sqrt(np.mean(mean_free * mean_free,
                              axis=self.axis, keepdims=True))
        return (mean, mean_free, std), mean_free / std

    def backward(self, state, root_gradients):
        mean, mean_free, std = state
        N = root_gradients[0].shape[self.axis]
        std2 = std * std
        tmp = root_gradients[0] / std2
        grad_arg = tmp - np.sum(tmp / N, axis=self.axis, keepdims=True)
        root_gradients = grad_arg
        return root_gradients

    def infer_outputs(self):
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]


def my_sequence_normalisation(arg1):
    return user_function(MySequenceNormalisation(arg1))


if __name__ == '__main__':

    #y = user_function(MySequenceNormalisation(x))
    #y = user_function(MySequenceNormalisation(x))  # my_sequence_mean(x)

    x1 = np.random.uniform(0.1, 1, size=(1, 4, 1)).astype(np.float32)
    gx1 = np.random.normal(0.1, 1, size=(1, 4, 1)).astype(np.float32)
    # gx1 = np.array([1, 2, 3, 4])[None, :, None]
    gx1 = np.abs(gx1)
    
    x = cntk.sequence.input(shape=(1), name='in_data', needs_gradient=True)

    y_my = user_function(MySequenceNormalisation(x, axis=-2))
    y_cn = sequence_normalisation()(x)

    test_numerical_grad(y_cn, {x: x1}, gx1, rtol=1e-4, atol=1e-4)
    test_numerical_grad(y_my, {x: x1}, gx1, rtol=1e-4, atol=1e-4)
    
    y_my.eval({x: x1})
    y_cn.eval({x: x1})

    from nt.utils.timer import TimerDict

    t = TimerDict()

    with t['my']:
        y_my.grad({x: x1})
    with t['cn']:
        y_cn.grad({x: x1})

    # %timeit y_my.grad({x: x1, c: c1})
    # %timeit y_cn.grad({x: x1, c: c1})

    print(t.as_yaml)
