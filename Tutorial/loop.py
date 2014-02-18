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
import theano
import theano.tensor as T

# 必要のない警告を抑制する設定
theano.config.warn.subtensor_merge_bug = False

print("kは整数スカラー、Aはベクトル(精度は未指定)")
k = T.iscalar("k")
A = T.vector("A")
print("pp(k) = %s, pp(A) = %s\n" % (pp(k), pp(A)))

#中間結果の計算に用いる関数
def inner_fct(prior_result, B):
    return prior_result * B

# scan()を用いて各ステップでの結果を計算
result, updates = theano.scan(fn=inner_fct,
                            outputs_info=T.ones_like(A),
                            non_sequences=A, n_steps=k)

# Scan has provided us with A ** 1 through A ** k.  Keep only the last
# value. Scan notices this and does not waste memory saving them.
final_result = result[-1]

power = theano.function(inputs=[A, k], outputs=final_result,
                      updates=updates)

print power(range(10),2)
#[  0.   1.   4.   9.  16.  25.  36.  49.  64.  81.]
