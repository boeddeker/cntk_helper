import numpy as np

import cntk
import cntk.layers

from cntk_helper.ops import sequence_normalisation, blstm, bce

model = cntk.layers.Sequential([
    sequence_normalisation(name='seq_norm'),
    blstm(513, name='blstm'),  # output size 1026
    sequence_normalisation(name='seq_norm'),
    cntk.layers.Dense(513, activation=None, name='dense'),
    sequence_normalisation(name='seq_norm'),
    cntk.relu,
    cntk.layers.Dense(513, activation=None, name='dense'),
    sequence_normalisation(name='seq_norm'),
    cntk.relu,
    cntk.layers.Dense(513 * 2, activation=cntk.sigmoid),
])

in_v = cntk.input_variable((513), np.float32, name='observation')

out_v = model(in_v)

c_mask_x = cntk.input_variable(
    (513), np.float32, name='oracle_binary_speech_mask')
c_mask_n = cntk.input_variable(
    (513), np.float32, name='oracle_binary_noise_mask')

c_mask = cntk.ops.splice(c_mask_x, c_mask_n)

# loss = cntk.squared_error(out_v, out_ref_v)
loss = bce(out_v, c_mask)
# eval_loss = cntk.binary_cross_entropy(out_v, c_mask)
eval_loss = bce(out_v, c_mask)

learning_rate = 0.001

momentum = cntk.momentum_as_time_constant_schedule(0.9)

lr_schedule = cntk.learning_rate_schedule(
    learning_rate, cntk.UnitType.minibatch)
# , l1_regularization_weight=1e-3
learner = cntk.adam(out_v.parameters, lr_schedule, momentum=momentum)

# tensorboard_writer = cntk.logging.TensorBoardProgressWriter(
#     freq=10, log_dir='/net/vol/boeddeker/tensorboard_log', model=loss)

trainer = cntk.Trainer(out_v, loss, learner)



t = TimerDict()
#iterator = tqdm_notebook(ds[:100], desc='')

for i, batch in enumerate(t['data load'](ds)):
    

    with t['preprocess ']:
        x, n, y = batch.X, batch.N, batch.observed
        X, N, Y = stft_v2([x, n, y]) 
        mask_X, mask_N = biased_binary_mask([X, N])

    with t['train']:
        trainer.train_minibatch({
                                in_v: np.abs(Y).astype(np.float32), 
                                c_mask_x: mask_X.astype(np.float32),
                                c_mask_n: mask_N.astype(np.float32)
                            })
    
    with t['display']:
        clear_output(wait=True)
        display(i)
        display(trainer.previous_minibatch_loss_average)
        # iterator.set_description(str())

    
batch = ds[-1]
    
x, n, y = batch.X, batch.N, batch.observed

X, N, Y = stft_v2([x, n, y]) 
mask_X, mask_N = biased_binary_mask([X, N])

eval_loss.eval({
                        in_v: np.abs(Y).astype(np.float32), 
                        c_mask_x: mask_X.astype(np.float32),
                        c_mask_n: mask_N.astype(np.float32)
                    })
    
#trainer.test_minibatch({
#                        in_v : in_d, 
#                        out_ref_v : out_ref_d,
#                    })


#np.array([trainer.model.eval({in_v : in_d})[:, :, 0], out_ref_d]).T
