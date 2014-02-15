import theano.tensor as T
import theano
import numpy as np

def report(x):
    print "--- attributes of the object %s ---" % x.name
    print "object:", x
    print "name:", x.name
    print "type:", x.type
    print "dtype:", x.dtype
    print "broadcastable dims:", x.broadcastable
    
print 'floatマトリックス'
x = T.fmatrix()
report(x)

print '名前"myvar"を持つ32-bit 整数スカラー'
x = T.scalar('myvar', dtype='int32')
report(x)

print '名前"myvar"を持つ32-bit 整数スカラー'
x = T.iscalar('myvar')
report(x)

# the same with above. broadcastable pattern indicats dimension of the variable.
# True means the length of the axis for that dimension is 1.
# empty list is a special case to mean scalar.
# pattern 	interpretation
# [] 	scalar
# [True] 	1D scalar (vector of length 1)
# [True, True] 	2D scalar (1x1 matrix)
# [False] 	vector
# [False, False] 	matrix
# [False] * n 	nD tensor
# [True, False] 	row (1xN matrix)
# [False, True] 	column (Mx1 matrix)
# [False, True, False] 	A Mx1xP tensor (a)
# [True, False, False] 	A 1xNxP tensor (b)
# [False, False, False] 	A MxNxP tensor (pattern of a + b)
x = T.TensorType(dtype='int32', broadcastable=())('myvar')

# config dependent float type (config.floatX is float 64 by default on x86_64)
x = T.scalar(name='x', dtype=T.config.floatX)
report(x)

# 1-dimensional vector (ndarray).
v = T.vector(dtype=T.config.floatX, name='v')
report(v)

# 2-dimensional ndarray in which the number of rows is guaranteed to be 1.
v = T.row(name=None, dtype=T.config.floatX)
report(v)

# 2-dimensional ndarray in which the number of columns is guaranteed to be 1.
v = T.col(name=None, dtype=T.config.floatX)
report(v)

# 2-dimensional ndarray
v = T.matrix(name=None, dtype=T.config.floatX)
report(v)

# 3-dimensional ndarray
v = T.tensor3(name=None, dtype=T.config.floatX)
report(v)

# 4-dimensional ndarray
v = T.tensor4(name=None, dtype=T.config.floatX)
report(v)

# constructors with fixed data type. (examples with tensor4)
# b: byte, w: word(16bit), l: int64, i: int32
# d:float64, f: float32, c: complex64, z: complex128
v = T.btensor4(name='v')
report(v)

v = T.wtensor4(name='v')
report(v)

v = T.itensor4(name='v')
report(v)

v = T.ltensor4(name='v')
report(v)

v = T.dtensor4(name='v')
report(v)

v = T.ftensor4(name='v')
report(v)

v = T.ctensor4(name='v')
report(v)

v = T.ztensor4(name='v')
report(v)

# you can of course define custom data type out of Theano tensors.
dtensor5 = T.TensorType('float64', (False,)*5)
x = dtensor5()
z = dtensor5('z')
report(x)
report(z)

# create shared variable from numpy matrix
# "shared" means value of this variable is shared among functions that call it.
# so to understand it throughly, you need to lean abut Theano functions.
a = np.random.randn(3, 4)
print "a =", a
x = theano.shared(a, name='x')
print "x = ", x, x.get_value()

# non-theano variable a can be converted to theano variable y explicitly.
# however you should not need to use this explicitly, since python/numpy numbers
# are converted to Theano tensor on the fly as needed.

# matrix example
y = T.as_tensor_variable(a, name='y')
report(y)
print "value of y =", y.value

# scalar example
y = T.as_tensor_variable(1.0, name='y')
report(y)
print "value of y =", y.value



