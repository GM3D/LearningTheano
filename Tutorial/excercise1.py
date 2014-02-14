import theano
import theano.tensor as T
# a = theano.tensor.vector() # declare variable
# out = a + a ** 10               # build symbolic expression
# f = theano.function([a], out)   # compile function
# print f([0, 1, 2])  # prints `array([0, 2, 1026])`

# Modify and execute this code to compute this expression: a ** 2 + b ** 2 + 2 * a * b.

a = T.vector()
b = T.vector()

out = a ** 2 + b ** 2 + 2 * a * b

f = theano.function([a, b], out)

print(f([0, 1, 2], [1, 2, 3]))
