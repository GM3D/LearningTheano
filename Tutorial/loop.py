# coding: UTF-8
# ループ
# 
# Theanoにはループを実行するためのスキャン関数scanが用意されている。
# スキャンには次のような特徴がある。
# 一般的な繰り返し操作に用いることができ、その特別なケースとしてループがある。
# 特別なケースとして、リダクションやマップ操作も扱うことができる。
# 与えられた入力シーケンスに対して、各タイムステップで出力を生成しながら
# 関数をスキャンしていくことになる。
# あるステップで関数の値を求めるのに、それ以前のK個までの結果を利用する
# ことができる。
# 一例として、sum()は初期値z = 0に対してz + x(i)という関数をスキャンすること
# で実現できる。
# for文をスキャンで置き換えられるケースも多く、Theanoでループに最も近い概念を
# 提供するのがスキャンである。
# 通常のループ構文に対してスキャンを用いる利点は:
#  繰り返しの回数もTheanoのグラフの一部として組み込める
#  GPUメモリに対する転送を極小化できる
#  微分を逐次的なステップで求めることができる
#  コンパイルされるので、ループ自体も通常の構文より若干高速化される
#  実際に必要とされる中間変数などに使用するメモリ量を自動検出することで、
#  メモリの全体的な使用量を抑えることができる。

from __future__ import print_function
from theano import pp
import theano
import theano.tensor as T
import numpy

# 必要のない警告を抑制する設定
theano.config.warn.subtensor_merge_bug = False

print("kは整数スカラー、Aはベクトル(精度は未指定)")
k = T.iscalar("k")
A = T.vector("A")
print("pp(k) = %s, pp(A) = %s\n" % (pp(k), pp(A)))

# 中間結果の計算に用いる関数
# scanの呼び出し時には最初の引数prior_resultには前回の計算結果が入る。
# 2番目の引数Bは変化しない。
# (下記参照)
def inner_fct(prior_result, B):
    return prior_result * B


# scan()を用いて各ステップでの結果を計算
# scanに与える関数の引数には一定の規約がある。すなわち、
# sequence引数のリスト、前回の計算結果、non-sequence引数のリストの順
# sequence引数: ループ毎に違う入力を外部から与える引数。
# キーワード引数sequencesで、一つの引数につきループのi=0、i=1、... に
# 対応したリストで指定する。
# 複数の引数に対して与えるリストの長さが異なる場合、一番短いリストの
# 分だけループが実行される。
# 前回の計算結果: これを使用するかどうかは、キーワード引数outputs_infoに
# よって指定される。outputs_infoがNoneの場合、前回の計算結果は引数に
# 渡されない。outputs_infoをNone以外に指定する場合は、関数の返り値と
# 一致する型情報、精度を持った値を指定しなければならない。
# non-sequence引数: キーワード引数non-sequencesによって指定する。
# ループ毎に変化しない引数。
# 各部分は必要がなければ空であってよい。
# この例では、各ステップ毎に変化する入力値は、前回の計算結果のみである。
# したがって、sequencesを指定しないことにより、デフォルトのNoneに設定している。
# また、outputs_info引数によって、計算結果の初期値を指定している。
# outputs_infoを指定した場合、関数が前回の計算結果を利用することを宣言した
# ことになり、したがって、今sequence=Noneであるため、第一引数が前回の
# 計算結果を受け取る引数と解釈される。
# 第2引数はnon-sequence、すなわちステップ毎に変化しない入力と解釈され、
# その値として上で定義した行列Aが渡される。

result, updates = theano.scan(fn=inner_fct,
                            outputs_info=T.ones_like(A),
                            non_sequences=A, n_steps=k)

# scanの返り値は、ループ毎の関数の返り値からなるリストと、theano.function
# に与えるために使用できるupdatesの辞書である。
# 今の場合、最後のループにおける計算結果のみが必要なので、リストの最後の
# 要素を取り出す。
final_result = result[-1]

# final_resultは間接的(プロシージャル)にA、kから定義されているので
# theano.functionに使用することができる
power = theano.function(inputs=[A, k], outputs=final_result,
                      updates=updates)

print("power(range(10), 2) = %s\n" %  power(range(10),2))
print("一致すべき値 = ")
print([0., 1., 4., 9., 16., 25., 36., 49., 64., 81.])

# テンソルの最初の軸について反復する例: 多項式の計算

# pythonの基本構文 for x in list と似たような働きで、スキャンを用いて
# テンソルの最初の軸について反復計算を行うことができる。

# 最大の係数の数は10000
max_coefficients_supported = 10000

print("係数ベクトルcoefficientsの定義")
coefficients = theano.tensor.vector("coefficients")
print("pp(coefficients) = %s\n" % pp(coefficients))

print("変数ベクトルxの定義")
x = T.scalar("x")
print("pp(x) = %s\n" % pp(x))


# 最大次数までの次数のリスト
full_range=theano.tensor.arange(max_coefficients_supported)

# componentsはscanによって生成される、多項式の各項からなるリスト。
# lambda関数では係数coefficient、次数powerのxを変数とする単項式が生成される。
# scanへの引数の与え方によって、関数の引数の解釈が決定される。
# outputs_info=Noneであるから、前回の計算結果を受け取る引数は存在しない。
# 前2つの引数coefficient, powerがsequence引数、3番目のxがnon_sequence引数となる。
# coefficientには後で具体的な値が決まるベクトル、powerには0次から最大次数-1までの
# 整数値が順に与えられる。
components, updates = theano.scan(fn=lambda coeff, power, free_var:
                                      coeff * (free_var ** power),
                                  outputs_info=None,
                                  sequences=[coefficients, full_range],
                                  non_sequences=x)

# 各項の和を取ることで多項式の値が計算できる。
# なお、前回の計算結果を用いて項を積算するようにすればここで和をとる必要は
# なくなるが、この例ではscanでは各項を個別に計算している。
polynomial = components.sum()

# 次数は最大次数まで自動生成が可能だが、coefficientsの方が長さが短い。
# この場合、短い方の長さで計算は打ち切られる。したがってn_stepsも指定する
# 必要がない。

# 最終的な関数の定義
calculate_polynomial = theano.function(inputs=[coefficients, x],
                                     outputs=polynomial)

test_coeff = numpy.asarray([1, 0, 2], dtype=numpy.float32)
print("calculate_polynomial(test_coeff, 3) = %f" 
      % calculate_polynomial(test_coeff, 3))

# Excercise
# 和をscanでとるように変更する。

def term_sum(coeff, power, prior_sum, x):
    return prior_sum + coeff * (x ** power)

initial_zero = numpy.asarray(0, dtype=numpy.float64)

partial_sum, updates = theano.scan(term_sum,
                                   outputs_info = initial_zero,
                                   sequences = [coefficients, full_range],
                                   non_sequences=x)

polynomial2 = partial_sum[-1]

calculate_polynomial2 = theano.function(inputs=[coefficients, x],
                                        outputs=polynomial2)

print("calculate_polynomial2(test_coeff, 3) = %f" 
      % calculate_polynomial2(test_coeff, 3))
