# encoding: UTF-8
from __future__ import print_function

import theano.tensor as T
from theano import pp, function, scan
import theano
import numpy

theano.config.warn.subtensor_merge_bug = False

# for文による通常のループ構文

# result = 1
# for i in xrange(k):
#     result = result * A

print("簡単な累積的計算の例: ベクトルA、整数kに対して、A**kを計算する")

k = T.iscalar("k")
A = T.vector("A")
print("pp(k) = %s, pp(A) = %s\n" % (pp(k), pp(A)))

def f(prior_result, A):
    print("prior_result = %s" % pp(prior_result))
    return prior_result * A

# scanによる漸化的なresultの定義
# scanに与える関数の引数には一定の規約がある。すなわち、
# sequence引数のリスト、前回の計算結果、non-sequence引数のリスト
# でなければならない。ただし各部分は必要がなければ空であってよい。
# この例では、各ステップ毎に変化する入力値は、前回の計算結果のみである。
# したがって、sequenceを指定しないことにより、デフォルトのNoneに設定している。
# また、outputs_info引数によって、計算結果の初期値を指定している。
# outputs_infoを指定した場合、関数が前回の計算結果を利用することを宣言した
# ことになり、したがって、今sequence=Noneであるため、第一引数が前回の
# 計算結果を受け取る引数と解釈される。
# 第2引数はnon-sequence、すなわちステップ毎に変化しない入力と解釈され、
# その値として上で定義した行列Aが渡される。
# n_stepsは繰り返しの回数を指定し、ここではkで与えられる。
result, updates = scan(fn = f,
                       outputs_info=T.ones_like(A),
                       non_sequences=A,
                       n_steps=k)

# scanから返される計算結果は、(i=0, i=1, ... i=k)に対するfnの返り値result
# 及び、updatesの辞書(この例では空辞書)からなるtupleである。
# updatesは、theano.functionに与えて使用することを前提とした内部状態の
# 更新を表すリスト。
# result最後の要素を取ることにより最終結果であるA**kが得られる。
# 必要ない途中の結果はscanにより自動的に開放される。

final_result = result[-1]

# final_resultは間接的(プロシージャル)にA、kから定義されているので
# theano.functionに使用することができる
power = function(inputs=[A,k], outputs=final_result, updates=updates)

# print("power = %s" % pp(power.maker.fgraph.outputs[0]))
# 長くなるので表示しない方が良い

print("power(range(10),2) = %s" % power(range(10),2))
print("power(range(10),4) = %s" % power(range(10),4))

# テンソルの最初の軸について反復する例: 多項式の計算

# pythonの基本構文 for x in list と似たような働きで、スキャンを用いて
# テンソルの最初の軸について反復計算を行うことができる。

print("係数ベクトルcoefficientsの定義")
coefficients = theano.tensor.vector("coefficients")
print("pp(coefficients) = %s\n" % pp(coefficients))

print("変数ベクトルxの定義")
x = T.scalar("x")
print("pp(x) = %s\n" % pp(x))

# 最大の係数の数は10000
max_coeffs = 10000

# Generate the components of the polynomial

# 関数termは係数coefficient、次数powerのxを変数とする単項式である。
def term(coefficient, power, x):
    return coefficient * (x ** power)

# componentsはscanによって生成される、多項式の各項からなるリスト。
# scanへの引数の与え方によって、termの引数の解釈が決定される。
# outputs_info=Noneであるから、前回の計算結果を受け取る引数は存在しない。
# 前2つの引数coefficient, powerがsequence引数、3番目のxがnon_sequence引数となる。
# coefficientには後で具体的な値が決まるベクトル、powerには0次から最大次数-1までの
# 整数値が順に与えられる。
components, updates = scan(fn=term,
                           outputs_info=None,
                           sequences=[coefficients, \
                                          theano.tensor.arange(max_coeffs)],
                           non_sequences=x)

# 各項の和を取ることで多項式の値が計算できる。
# なお、前回の計算結果を用いて項を積算するようにすればここで和をとる必要は
# なくなるが、この例ではscanでは各項を個別に計算している。
polynomial = components.sum()

# 次数はmax_coeffsまで自動生成が可能だが、coefficientsの方が長さが短い。
# この場合、短い方の長さで計算は打ち切られる。したがってn_stepsも指定する
# 必要がない。
# また自動生成されるシーケンスはsequencesに記述しないことに注意。

# 最終的な関数の定義
calculate_polynomial = theano.function(inputs=[coefficients, x], outputs=polynomial)

print("入力として与える定数ベクトルはnumpy.arrayとして与える。")
test_coefficients = numpy.asarray([1, 0, 2], dtype=numpy.float32)
print("test_coefficients = %s\n" % test_coefficients)
test_x = 3
print("test_x = %s\n" % test_x)

print("calculate_polynomial(test_coefficients, test_x) = %s" % 
      calculate_polynomial(test_coefficients, test_x))
print("一致すべき結果: %f" % 
      (1.0 * (3 ** 0) + 0.0 * (3 ** 1) + 2.0 * (3 ** 2),))
