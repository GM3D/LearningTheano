# encoding: UTF-8

import theano.tensor as T
from theano import function
import numpy

x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y

x = T.dmatrix('x')
y = x * 2
print y.owner.op.name

# inputs[0] = x, imputs[1] = DimShuffle{x, x}.0 
# DimShuffle{x, x}は行と列サイズを固定しない行列を表す
# 2がDimShuffle{x, x}によってブロードキャスティングされている
print y.owner.inputs

# そのため両方のinputが同じタイプになって乗算可能になっている
print map(type, y.owner.inputs)

f = function([x], y)

# 行列とスカラーの積はブロードキャスティングによって成分毎に計算される
print f(numpy.asarray([[1, 2], [3, 4]]))

# 実はy.owner.inputs[1]のownerはtheano.gof.graph.Applyクラスである
print "type(y.owner.inputs[1].owner) =", type(y.owner.inputs[1].owner)

# そのApplyノードに接続されているopは
# theano.tensor.elemwise.DimShuffleオブジェクトであり、
print "type(y.owner.inputs[1].owner.op)=", type(y.owner.inputs[1].owner.op)

# Applyノードへの入力が定数2.0となっている。
print "y.owner.inputs[1].owner.inputs = ", y.owner.inputs[1].owner.inputs
