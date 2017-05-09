import numpy as np

import cntk
from cntk import output_variable, user_function
from cntk.ops.functions import UserFunction

# ToDo: my_sequence_sum does not work


def sequence_mean(x, keepdims=False):
    sequence_length = cntk.sequence.reduce_sum(cntk.reduce_sum(x) * 0 + 1)
    mean = cntk.sequence.reduce_sum(x) / sequence_length
    if keepdims:
        return cntk.sequence.broadcast_as(mean, x)
    else:
        return mean


class MySequenceSum(UserFunction):

    def __init__(self, arg1, axis=0, name='f1'):
        super().__init__([arg1], name=name)
        self.axis = axis

    def forward(self, arguments, device=None, outputs_to_retain=None):
        mean = np.sum(arguments[0], axis=self.axis, keepdims=False)
        print(arguments[0].shape, mean.shape)
        return None, mean

    def backward(self, state, root_gradients):
        raise NotImplementedError()

    def infer_outputs(self):
        return [output_variable(
            self.inputs[0].shape,
            self.inputs[0].dtype,
            self.inputs[0].dynamic_axes[:-1]
        )]


def my_sequence_sum(arg1, axis):
    return user_function(MySequenceSum(arg1, axis))

if __name__ == '__main__':

    x0 = np.random.normal(size=(1, 4, 3)).astype(np.float32)
    x = cntk.sequence.input(shape=(3), name='in_data', needs_gradient=True)

    x_sum_cn = cntk.sequence.reduce_sum(x)
    x_sum_my = my_sequence_sum(x, axis=-2)

    print(x)
    print(x_sum_cn)
    print(x_sum_my)
    x_sum_cn.eval({x: x0})
    x_sum_my.eval({x: x0})
