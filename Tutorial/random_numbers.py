# coding: UTF-8
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

# RandomStreamsクラスはソースとなる乱数ストリーム
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

# borrow=Trueフラグについての詳細は、
# http://deeplearning.net/software/theano/tutorial/aliasing.html
# を参照。
# 基本的には、返り値rng_valはrng内部の状態変数のコピーではなく参照となる。
# Theanoで定義されるオブジェクトのget_value()、set_value()メソッド及び、
# shared変数のコンストラクタで指定できる。
rng_val = rv_u.rng.get_value(borrow=True)

# rng_valにseedを設定
rng_val.seed(89234)
         
# rngに変更されたrng_valをセットする
rv_u.rng.set_value(rng_val, borrow=True)

# srngにseedを設定すると、そこから乱数でrv_uとrv_nの双方に違うseedが与えられる
srng.seed(902340)

state_after_v0 = rv_u.rng.get_value().get_state()
print state_after_v0

# rv_uの生成器の状態が更新される
print nearly_zeros()       

# rv_uの値をチェック
v1 = f()
print v1

# nearly_zeros()の呼び出し前の状態を復元する
rng = rv_u.rng.get_value(borrow=True)
rng.set_state(state_after_v0)
rv_u.rng.set_value(rng, borrow=True)

#                                      v1
#                                       |
# ---state_after_v0-->nearly_zeros()--->f()
#         |
#         -------------->f()----------->f()
#                         |             |
#                        v2             v3

#v2 != v1
v2 = f()

# v3 == v1
v3 = f()

print "v1 == v3:", v1 == v3

from theano.sandbox.rng_mrg import MRG_RandomStreams

class Graph():
    def __init__(self, seed=123):
        self.rng = RandomStreams(seed)
        self.y = self.rng.uniform(size=(1,))

g1 = Graph(seed=123)
f1 = function([], g1.y)

g2 = Graph(seed=987)
f2 = function([], g2.y)

print '# 通常はf1とf2はそれぞれ独立'
print 'f1() = ', f1()
print 'f2() = ', f2()

def copy_random_state(g1, g2):
    """g1からg2にRNGの状態をコピーする。
    MRG_RandomStreamsはrstateアトリビュートに現在の状態を保持している。
    また、state_updatesはこのストリームから生成される任意の乱数発生器に対する、
    状態変数とどのような分布の乱数を生成するかのtupleからなるリストである。
    この関数は、g1とg2から派生している乱数生成器の種類と個数は同じであり、
    両者の内部状態変数のみが違う場合に正しく動作する。
    これは例えば、データベースから読み込んだ旧乱数オブジェクトの状態を
    新しく生成したオブジェクトにロードする場合等に相当する。
    """
    if isinstance(g1.rng, MRG_RandomStreams):
        g2.rng.rstate = g1.rng.rstate
    print "g1.rng.state_updates =", g1.rng.state_updates
    print "g2.rng.state_updates =", g2.rng.state_updates
    print "copying from g1 to g2..."
    i = 0
    for (su1, su2) in zip(g1.rng.state_updates, g2.rng.state_updates):
        value1 = su1[0].get_value()
        value2 = su2[0].get_value()
        print "value of g1.rng.state_updates[%d] = " % i, value1
        print "value of g2.rng.state_updates[%d] = " % i, value2
        su2[0].set_value(value1)
        i += 1

print '# g1の状態をg2にコピー'
copy_random_state(g1, g2)

print '# 以降は同期した値が生成される'
print 'f1() = ', f1()
print 'f2() = ', f2()

# ロジスティック回帰 (Logistic Regression)
import numpy
import theano

# ここでは初期にに乱数を使用するだけなので、TheanoのRandomStreamは必要ない
rng = numpy.random

# データの個数
N = 400
# 特徴点(入力ノード)の数、28*28 = 784 (MNIST手書き文字データ)
feats = 784

# 入力データ（乱数による生成）
# D[0]: feats個の乱数(-1～1)を一つのデータとし、それをN個用意する。
# D[1]: N個のデータそれぞれについて、"1"であるか"0"であるかの「正解」
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))

# 学習させる回数
training_steps = 10000

# Theanoのシンボル変数を定義
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")
print "学習前のネットワーク:"
print w.get_value(), b.get_value()

# 数式グラフの構築
# xで表されるターゲットデータが分類1に属する確率
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))

# ターゲットデータが1であるか0であるかの判定に用いる式
# 確立がこの閾値越えれば1であると判定する
prediction = p_1 > 0.5

# 各学習データxがそれぞれ属する値からなるベクトルをyとしたとき、
# 際尤度関数の対数を取ったもの。(相互エントロピー損失関数)
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)

# 学習により最小化すべき量。xentの平均値に、安定化のための項w^2を加えてある。
cost = xent.mean() + 0.01 * (w ** 2).sum()

# w, bによるcostの微分係数。
gw, gb = T.grad(cost, [w, b])

# 実行時関数の定義、コンパイル。
# 教師データx, yを与えて、wとbをここではシンプルな最急降下法で求める。
# 一層の線形ネットワークなので最急降下法で十分である。
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))

# 学習が済んだネットワークの評価に用いる。
predict = theano.function(inputs=[x], outputs=prediction)

# 学習
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print "学習後のネットワーク:"
print w.get_value(), b.get_value()

PD0 = predict(D[0])
print "教師データの正解:", D[1]
print "教師データについてのネットワークによる判定結果:", PD0

correct = 0
wrong = 0
for inout in zip(D[1], PD0):
    if inout[0] == inout[1]:
        correct += 1
    else:
        wrong += 1

total = correct + wrong
print "教師データ個数 %d個、正解 %d個、不正解 %d個" % (total, correct, wrong)
ratio = float(correct) / float(total)
print "正解率 = %f" % ratio
