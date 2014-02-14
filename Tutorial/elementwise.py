import theano.tensor as T
from theano import *

# x is a double precision matrix.
x = T.dmatrix('x')

# s is logistic function applied emementwise to x.
s = 1 / (1 + T.exp(-x))

# compile s
logistic = function([x], s)

# execute it.
print logistic([[0, 1], [-1, -2]])
