# coding: UTF-8

import theano.tensor as T
from theano import function, shared

count = shared(0)
inc = T.iscalar('inc')
accum = function([inc], count, updates=[(count, count + inc)])

#initial value is 0
print count.get_value()

# increment it by 1. current value 0 is printed.
print accum(1)

# check the new value which is 1.
print count.get_value()

# increment by 300. current value 1 is printed.
print accum(300)

# print the new value 301
print count.get_value()

#set counter to -1
count.set_value(-1)

# increment by 3, current value -1 is printed
print accum(3)

# counter's new value is -1 + 3 = 2
print count.get_value()

# countを共有する関数decrを定義
decr = function([inc], count, updates=[(count, count - inc)])

#現在の値2が表示される
print decr(2)

# 2 - 2 = 0 が表示される
print count.get_value()

# countを用いた別の関数を定義
fn_of_count = count * 2 + inc

# 一時的にcountの代わりに与える変数として、countと同じ型の変数を用意する
foo = T.scalar(dtype=count.dtype)

# givens=(count, foo)によって、countの代わりにfooを用いることを指定する
skip_shared = function([inc, foo], fn_of_count, givens=[(count, foo)])

# countの代わりにfoo = 3が使用される (3 * 2 + 1 = 7が表示される)
print skip_shared(1, 3)

#countの値は変更されない (0のまま)
print count.get_value()

# 複数のgivensを使用するときは、お互いに独立なものに限ること。
# 評価順序が規定されていないので、依存性のある物を複数用いると結果が不定になる。

