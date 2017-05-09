import cntk
import numpy as np


def log2(x):
    return cntk.ops.log(x) / np.log(2)


def blstm(N, name=''):

    #@cntk.layers.blocks.BlockFunction('BidirectionalLongShortTermMemory', name)
    # def blstm(x):

    return cntk.layers.Sequential(
        [
            (
                # first tuple entry: forward pass
                cntk.layers.Recurrence(cntk.layers.LSTM(N)),
                cntk.layers.Recurrence(cntk.layers.LSTM(
                    N), go_backwards=True)    # second: backward pass
            ),
            cntk.layers.splice
        ], name=name
    )

    # f = cntk.layers.Recurrence(C.layers.LSTM(N))(x)
    # b = cntk.layers.Recurrence(C.layers.LSTM(N))(x, go_backwards=False)

def bce(x, t):
    return -sequence_mean(cntk.reduce_mean(
        t * log2(x + 1e-6) + (1 - t) * log2((1 - x) + 1e-6)))
