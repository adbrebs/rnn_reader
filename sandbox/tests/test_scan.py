
import theano
import theano.tensor as T

def step(h_pre, covariance_pre):
    h = 2 * h_pre
    inc_covariance = h.dimshuffle((0, 'x', 1)) * h.dimshuffle((0, 1, 'x'))
    return h, covariance_pre + inc_covariance

init_state = T.alloc(.0, 32, 256)

(seq_h, seq_cov), _ = theano.scan(
    fn=step, outputs_info=[init_state, T.alloc(.0, 32, 256, 256)],
    n_steps=1000)

cost = seq_h[-1].sum()
g = T.grad(cost, init_state)

f = theano.function([], g)
print 'compiled'
for i in range(1):
    f()

print 'all good'