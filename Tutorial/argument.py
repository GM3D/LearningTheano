import theano.tensor as T
from theano import *

# x and y are double precision scalars defined at once.
x, y = T.dscalars('x', 'y')

# define sum of those
z = x + y

# define default value for y with Param object.
f = function([x, Param(y, default=1)], z)

# works. should be 34
print f(33)

# override default value, should be 35
print f(33, 2)

# define three double precision scalars
x, y, w = T.dscalars('x', 'y', 'w')

# define an expression out of them
z = (x + y) * w

# attach default values and name for overriding keyword arg
f = function([x, Param(y, default=1), Param(w, default=2, name='w_by_name')], z)

print f(33) # (33 + 1) * 2 = 68
print f(33, 2) # (33 + 2) * 2 = 70
print f(33, 0, 1) # (33 + 0) * 1 = 33
print f(33, w_by_name=1) # (33 + 1) * 1 = 34
print f(33, w_by_name=1, y=0) # (33 + 0) * 1 = 33
