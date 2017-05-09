import numpy as np

import cntk
from cntk import output_variable, user_function
from cntk.ops.functions import UserFunction

from cntk_helper.numerical_grad import test_numerical_grad

class ParameterizedVectorNorm(UserFunction):

    def __init__(self, arg1, axis=0, 
                 name='ParameterizedVectorNorm', eps=1e-6):
        super().__init__([arg1], name=name)
        self.axis = axis
        self.eps = eps

    def forward(self, arguments, device=None, outputs_to_retain=None):
        x = arguments[0]
        print(x.shape)
        b = np.sum(
            x * x,
            axis=self.axis,
            keepdims=True
        )
        
        v = np.sqrt(b + self.eps)

        return (x, b, v), x / v

    def backward(self, state, root_gradients):
        x, b, v = state
        
        J_y_star = root_gradients[0]

        q = np.sum(J_y_star * x, axis=self.axis, keepdims=True)

        J_x_star = (J_y_star - q * x / (b + self.eps)) / v

        return J_x_star

    def infer_outputs(self):
        shape = [*self.inputs[0].dynamic_axes, self.inputs[0].shape]
        shape[self.axis]
        
        #def translate(ax):
        #    if isinstance(ax, cntk.axis.Axis):
        
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]


def parameterized_vector_norm(arg1, axis=0, eps=1e-6):
    return user_function(ParameterizedVectorNorm(arg1, axis=axis, eps=eps))


if __name__ == '__main__':
    for axis in [0, 1, -1, -2]:

        x1 = np.random.uniform(0.1, 1, size=(1, 4, 1)).astype(np.float64)
        gx1 = np.random.normal(0.1, 1, size=(1, 4, 1)).astype(np.float64)
        
        x = cntk.sequence.input(shape=(1), name='in_data', 
                                needs_gradient=True, dtype=np.float64)
    
        # y_my = user_function(MySequenceNormalisation(x, axis=-2))
        y_my = parameterized_vector_norm(x, eps=0, axis=axis)
    
        # test_numerical_grad(y_cn, {x: x1}, gx1, rtol=1e-4, atol=1e-4)
        test_numerical_grad(y_my, {x: x1}, gx1, rtol=1e-4, atol=1e-4)
        
        y_my.eval({x: x1})
        # y_cn.eval({x: x1})
