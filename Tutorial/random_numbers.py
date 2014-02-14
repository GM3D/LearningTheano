# coding: UTF-8
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

# RandomStreamsクラスはソースとなる乱数生成器
srng = RandomStreams(seed=234)

# srngを元に、一様分布乱数生成器を作成(2 x 2マトリックス)
rv_u = srng.uniform((2,2))

# 同じソースから正規分布乱数を生成
rv_n = srng.normal((2,2))

# 実際に値を取り出せる関数をコンパイル。乱数なので入力は必要ない。
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)    #Not updating rv_n.rng

# 毎回違う値を生成
print f()
print f()

# 内部状態を更新しないので毎回同じ値を生成
print g()
print g()

# 一つの関数内では同じ乱数が複数回使われても、評価は一回だけ（同じ値）
# 浮動小数の丸め誤差を除いて、下の値は0になる。
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

print nearly_zeros()
