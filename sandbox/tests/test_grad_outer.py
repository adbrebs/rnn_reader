import theano
import theano.tensor as T

x = T.vector('x')

y = x * x.dimshuffle((0, 'x'))

g = theano.grad(y.sum(), x)

f = theano.function([x], g)

theano.printing.debugprint(f)