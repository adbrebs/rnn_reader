
import theano
import theano.tensor as T

def step(h_pre, covariance_pre):
    h = 2 * h_pre
    inc_covariance = h.dimshuffle((0, 'x', 1)) * h.dimshuffle((0, 1, 'x'))
    return h, covariance_pre + inc_covariance

(seq_h, seq_cov), _ = theano.scan(
    fn=step, outputs_info=[T.alloc(.0, 32, 256), T.alloc(.0, 32, 256, 256)],
    n_steps=10000)

f = theano.function([], seq_cov[-1])
print 'compiled'
for i in range(300):
    f()

print 'all good'