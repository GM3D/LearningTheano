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
print("一致すべき結果: %f\n" % 
      (1.0 * (3 ** 0) + 0.0 * (3 ** 1) + 2.0 * (3 ** 2),))

# N個の整数の総和
print("N個の整数の総和: up_toは整数スカラー、この値までの自然数の和を求める。")
up_to = T.iscalar("up_to")

# 積算を行う関数。sum_to_dateが前回までの和、arrange_valが今回加える値。
# arrange_valはsequences引数になり、sum_to_dateが前回の計算値なので
# 引数の順番はこのようになる。
def accumulate_by_adding(arange_val, sum_to_date):
    return sum_to_date + arange_val

# 和をとる数列は、0, 1, 2, ... up_to - 1
seq = T.arange(up_to)

# 注意!
# 初期値として以下を使用するとエラーになる。
# その理由は、特に指定しないと、定数0はその値を表現しうる最も小さな整数型int8に
# なり、それに応じて和もint8になる。一方、各ステップでarrange_valに与えられて
# 計算される値はint32型なので、暗黙にダウンキャストが行われることになる。
# このような場合、Theanoはメッセージを発して正しい初期値の型を要求する。
# outputs_info = T.as_tensor_variable(0)

# sequence引数と型をそろえてダウンキャストが起こらないようにする。
outputs_info = T.as_tensor_variable(numpy.asarray(0, seq.dtype))

# scan_updatesは必要ないが、scanの返り値として受け取る。
scan_result, scan_updates = theano.scan(fn=accumulate_by_adding,
                                        outputs_info=outputs_info,
                                        sequences=seq)

# scan_resultを出力と定義しているので、結果は最後の和だけでなく、途中経過も
# 含めたリストとなる。
triangular_sequence = theano.function(inputs=[up_to], outputs=scan_result)

# test
some_num = 10
print("triangular_sequence(%d) = %s\n" % 
      (some_num, triangular_sequence(some_num)))
print("一致すべき値")
print([n * (n + 1) // 2 for n in xrange(some_num)])
print('\n')

# 別の例
# 原型となる行列があり、各ステップではこの原型と同じ型、サイズの行列を
# 出力する。各成分は一つを除いて0であり、その一か所を指定する添字と
# 値はsequences引数で与えられる。

#各ステップで変更対象となる成分の位置を並べたNx2行列
location = T.imatrix("location")

# 変更対象となる成分を置き換える値
values = T.vector("values")

# 原型となる行列。この例では型とサイズだけが重要である。
output_model = T.matrix("output_model")


def set_value_at_position(a_location, a_value, output_model):
    """output_modelのa_locationで指定された成分をa_valueで置き換えた
    行列を返す。
    以下の実装では、zerosがTheanoのシンボリック変数なので、
    zeros_subtensorも、成分の「値」ではなく、成分そのものを示す
    ことに注意。
    """
    zeros = T.zeros_like(output_model)
    zeros_subtensor = zeros[a_location[0], a_location[1]]
    return T.set_subtensor(zeros_subtensor, a_value)

# 前回の結果を利用しないので、外部からoutputs_infoを与える必要はない。
# a_location、a_valueがsequence変数、output_modelがnon-sequenceとして
# 使用される。
result, updates = theano.scan(fn=set_value_at_position,
                              outputs_info=None,
                              sequences=[location, values],
                              non_sequences=output_model)

# 各ステップで生成された行列からなるリストを出力とする
assign_values_at_positions = theano.function(inputs=[location, values, output_model], outputs=result)

print("行列生成のテスト")

test_locations = numpy.asarray([[1, 1], [2, 3]], dtype=numpy.int32)
print("置き換える成分の位置リスト = %s\n" % test_locations)

test_values = numpy.asarray([42, 50], dtype=numpy.float32)
print("置き換え後の値リスト = %s\n" % test_values)

test_output_model = numpy.zeros((5, 5), dtype=numpy.float32)
print("原型となる行列:")
print(test_output_model)

print("置き換え結果 = \n%s\n" % assign_values_at_positions(test_locations, test_values, test_output_model))

# 過去の計算結果を複数参照する例: 想起ニューラルネット
# この例は実用的なものではなく、scanの使い方を理解するためのもの。
# 単に過去の入出力の履歴に現在の出力が依存するネットワークと
# とらえておけばよい。

# u[t]は外部から与えられる入力
# x[t]はネットワークの出力値
# yはx[t - 3]にW_outを演算して得られる。
# 表記法として、x_tm4はx[t - 4]を、y_tp1はy[t + 1]等を表す。

def oneStep(u_tm4, u_t,
            x_tm3, x_tm1, y_tm1,
            W, W_in_1, W_in_2,
            W_feedback, W_out):
    x_t = T.tanh(theano.dot(x_tm1, W) + \
                     theano.dot(u_t,   W_in_1) + \
                     theano.dot(u_tm4, W_in_2) + \
                     theano.dot(y_tm1, W_feedback))
    y_t = theano.dot(x_tm3, W_out)
    return [x_t, y_t]

# uはベクトルの時系列
u  = T.matrix()

# 初期値としてx[-3]とx[-1]が必要になるので、xの初期データx0は行列とする
x0 = T.matrix()

# yは初期値として必要なのはy[-1]だけなので、y0はベクトルでよい
y0 = T.vector()

W = T.vector()
W_in_1 = T.vector()
W_in_2 = T.vector()
W_feedback = T.vector()
W_out = T.vector()

# for second input y, scan adds -1 in output_taps by default
# ([x_vals, y_vals],updates) =\
#     theano.scan(fn = oneStep, \
#                     sequences    = dict(input = u, taps= [-4,-0]), \
#                     outputs_info = [dict(initial = x0, taps = [-3,-1]),y0], \
#                     non_sequences  = [W,W_in_1,W_in_2,W_feedback, W_out])

# Shared変数の利用 - ギッブスサンプリング


# W_values = T.dmatrix() 
# bvis_values = T.dvector()
# bhid_values = T.dvector()

# W = theano.shared(W_values) # we assume that ``W_values`` contains the
#                             # initial values of your weight matrix

# bvis = theano.shared(bvis_values)
# bhid = theano.shared(bhid_values)

# trng = T.shared_randomstreams.RandomStreams(1234)

# def OneStep(vsample) :
#    hmean = T.nnet.sigmoid(theano.dot(vsample, W) + bhid)
#    hsample = trng.binomial(size=hmean.shape, n=1, p=hmean)
#    vmean = T.nnet.sigmoid(theano.dot(hsample, W.T) + bvis)
#    return trng.binomial(size=vsample.shape, n=1, p=vmean,
#                         dtype=theano.config.floatX)

# sample = theano.tensor.vector()

# values, updates = theano.scan(OneStep, outputs_info=sample, n_steps=10)

# gibbs10 = theano.function([sample], values[-1], updates=updates)
