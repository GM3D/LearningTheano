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
print("xは倍精度小数ベクター、yはxの成分毎に2乗演算を適用したもの")

x = T.dvector('x')
y = x ** 2

print("pp(x) = %s, pp(y) = %s" % (pp(x), pp(y)))


# scanはTheanoでloopを行う演算子
# updatesに関する詳細はLoopの章で扱う

J, updates = scan(lambda i, y,x : T.grad(y[i], x), 
                  sequences=T.arange(y.shape[0]), 
                  non_sequences=[y,x])

f = function([x], J, updates=updates)

print("xを2次元ベクトルとして微分を計算、x = [4, 4]の点における値を求める")
print("f([4, 4]) =\n%s\n" % f([4, 4]))

# Hessianの計算
# Hessianはtheano.gradient.hessian()を用いることで計算できる。
# しかしここでは、理解を深めるために手動でHessianを求めてみる。

# Hessianは、スカラー関数fに対して、d^2f/dx[i]dx[j]で定義される行列である。
# したがって、本質的に、上記のヤコビアンの計算においてベクトル関数yを用いる
# 代わりに、スカラー関数fの一階微分df/dx[i]を用いればよい。

print("Hessianの計算")
x = T.dvector('x')
y = x ** 2
cost = y.sum()
# cost = x1^2 + x2^2 + ... + xn^2 のHessianを求める
gy = T.grad(cost, x)
H, updates = scan(lambda i, gy,x : T.grad(gy[i], x), 
                   sequences=T.arange(gy.shape[0]),
                   non_sequences=[gy, x])
f = function([x], H, updates=updates)
print("f([4, 4]) =\n %s\n" % f([4, 4]))

# ヤコビアンとベクトルの乗算を用いて記述されたアルゴリズムの場合、
# ヤコビアンを計算してからベクトルをそれに掛けるよりも効率の良い
# 計算法がしばしば存在する。その一例が
# Barak A. Pearlmutter, “Fast Exact Multiplication by the Hessian”, 
# Neural Computation, 1994
# で紹介されている方法である。
# 本来はTheanoが自動的にそのような最適化を行ってくれることが望ましいが、
# 実際にはそのような最適化は非常に困難なので、その代わりにそれをサポート
# するための演算子が用意されている。

# R演算子
# R演算子は、(df(x)/dx)*vのような、ヤコビアンとベクトルの積を計算するために
# 用意された演算子である。xは単にベクトルのみではなく、行列や、一般のテンソル
# であってもよい。そのような場合、積も一種のテンソル積になる。
# 実際にマシンラーニングに適用した場合、このような表式はしばしばネットワークの
# 重み行列による微分になるため、このように一般的な積をサポートできるように設計
# されている。

print("R演算子を用いた行列*ベクトルの計算")
W = T.dmatrix('W')
V = T.dmatrix('V')
x = T.dvector('x')
y = T.dot(x, W)
JV = T.Rop(y, W, V)
f = function([W, V, x], JV)
print("f([[1, 1], [1, 1]], [[2, 2], [2, 2]], [0,1]) = \n%s\n"
      % f([[1, 1], [1, 1]], [[2, 2], [2, 2]], [0,1]))

# 同様に、L演算子を用いるとベクトルv、x及びスカラー関数fに対して
#  v * df(x)/dx を計算できる。
print("L演算子を用いたベクトル*行列の計算")
W = T.dmatrix('W')
v = T.dvector('v')
x = T.dvector('x')
y = T.dot(x, W)
VJ = T.Lop(y, W, v)
f = function([v,x], VJ)
print("f([2, 2], [0, 1]) = \n%s\n" % f([2, 2], [0, 1]))

# R演算子を用いて実際にHessian*ベクトルを計算してみる
print("R演算子を用いずベクトル*Hessianを計算する")
x = T.dvector('x')
v = T.dvector('v')
y = T.sum(x ** 2)
gy = T.grad(y, x)
vH = T.grad(T.sum(gy * v), x)
f = function([x, v], vH)
print("f([4, 4], [2, 2]) = \n%s\n" % f([4, 4], [2, 2]))

print("R演算子を用いてのHessian*ベクトルの計算")
x = T.dvector('x')
v = T.dvector('v')
y = T.sum(x ** 2)
gy = T.grad(y, x)
Hv = T.Rop(gy, x, v)
f = function([x, v], Hv)
print("f([4, 4], [2, 2]) = \n%s\n" % f([4, 4], [2, 2]))
