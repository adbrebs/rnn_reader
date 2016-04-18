import theano
import theano.tensor as T


x = T.vector('x')
z = T.vector('z')

y1 = x * x.dimshuffle((0, 'x'))
y2 = T.outer(x, x)

g1 = theano.grad(T.dot(y1, z).sum(), x)
g2 = theano.grad(T.dot(y2, z).sum(), x)

f1 = theano.function([x, z], g1)
f2 = theano.function([x, z], g2)


theano.printing.debugprint(f1)
print '----------------------------------------'
theano.printing.debugprint(f2)
