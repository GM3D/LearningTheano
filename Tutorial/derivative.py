# encoding: UTF-8
from __future__ import print_function

import theano.tensor as T
from theano import pp, function, scan
from theano.printing import debugprint

# TheanoはGraphという形で数式がどのように構成されているかを情報として持っている。
# このため、出力から順に微分のチェインルールを適用することで、数式の入力変数による
# 微分を式として求めることができる。

# このため、原理的に未定義の関数や陰関数の微分はできない。
# あくまでTheanoの演算を用いて陽に構成された式が対象になる。
# この点がMaximaのような数式処理ソフトとは異なる。

print("倍精度スカラーとその2乗yを定義")
x = T.dscalar('x')
y = x ** 2

print("gyはyのxによる微分")
gy = T.grad(y, x)

print("コンパイル、最適化前のgyを表示")
print("pp(gy) = %s\n" % pp(gy))

'((fill((x ** 2), 1.0) * 2) * (x ** (2 - 1)))'
print("fill(x ** 2, 1.0)はx**2と同じ形のテンソル(ここではスカラー)で全成分が1.0")
print("つまり 1 * 2 * (x ** (2 - 1))で 2*xになっている。")

print("fはgyをコンパイル、最適化したもの. debugprintを見ると2*xになっていることが分かる。")
f = function([x], gy)
print(debugprint(f))

print("さらにfのmaker.fgraph.outputs[0]プロパティをpretty printしても分かる。")
print("pp(f.maker.fgraph.outputs[0]) = %s" % pp(f.maker.fgraph.outputs[0]))

print("f(4) = %f" % f(4))
# array(8.0)

print("f(94.2) = %f" % f(94.2))
# array(188.40000000000001)

print("xは倍精度行列")
x = T.dmatrix('x')
print("pp(x) = %s\n" % pp(x))

print("sはxのロジスティック関数")
s = T.sum(1 / (1 + T.exp(-x)))
print("pp(s) = %s\n" % pp(s))

print("gsはSのxに関する微分")
gs = T.grad(s, x)
print("pp(gs) = %s\n" % pp(gs))

print("dlogisticはgsのコンパイル、最適化形")
dlogistic = function([x], gs)
print("pp(dlogistic.maker.fgraph.outputs[0]) = %s\n" %
      pp(dlogistic.maker.fgraph.outputs[0]))

print("dlogistic([[0, 1], [-1, -2]]) =\n", dlogistic([[0, 1], [-1, -2]]))

# ヤコビアン
# n次元のベクトル変数xと、その関数であるm次元ベクトルyを考える。
# 典型的な座標変換などではn = mであり、その場合は、dy[i]/dx[j]を成分とする
# 行列を考えると、正方行列なので行列式を定義することができる。この行列式
# J = det(dy[i]/dx[j])をヤコビアンという。
# Theanoでは、これをより一般的に n != m の場合も考え、行列式をとる前の
# 一次微分係数 dy[i]/dx[j] をヤコビアンと呼んでいる。

# 微分演算を用いて直接ヤコビアンを求める場合
print("xは倍精度小数ベクター、yはx**2")

x = T.dvector('x')
y = x ** 2

print("pp(x) = %s, pp(y) = %s" % (pp(x), pp(y)))


J, updates = scan(lambda i, y,x : T.grad(y[i], x), 
                  sequences=T.arange(y.shape[0]), 
                  non_sequences=[y,x])
f = function([x], J, updates=updates)

print("f([4, 4]) =\n%s" % f([4, 4]))
