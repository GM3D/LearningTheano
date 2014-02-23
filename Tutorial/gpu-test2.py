from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], sandbox.cuda.basic_ops.gpu_from_host(T.exp(x)))
print f.maker.fgraph.toposort()
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()
print 'Looping %d times took' % iters, t1 - t0, 'seconds'
print 'Result is', r
print 'Numpy result is', numpy.asarray(r)
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print 'Used the cpu'
else:
    print 'Used the gpu'

実行例
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python check2.py
Using gpu device 0: GeForce GTX 580
[GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>)]
Looping 1000 times took 0.34898686409 seconds
Result is <CudaNdarray object at 0x6a7a5f0>
Numpy result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
  1.62323296]
Used the gpu

この例では、GPUからホストへのコピーをなくしただけでおよそ50%の速度向上
が得られた。それぞれのファンクションコールで返される結果はNumpy array
ではなく、CudaNdarrayである。これは通常のキャストの機構によってNumpy
ndarrayに変換することができる。

GPUを最大限に活かす
この例を最大限に速くするためには、返される出力をTheanoにコピーさせない
ため、borrwo=Trueフラグを立ててoutインスタンスを使用する必要がある。
これは、Theanoは内部的に使用するためにメモリを事前に割り当てる(ワーキ
ングバッファなどの目的で)が、この領域は通常結果を返すのには直接使われ
ず、関数の呼び出し毎に新しく割り当てられたメモリ領域に結果がコピーされ
て返されるためである。これはその後の関数呼び出しによって結果が破壊され
るのを防ぐためであり、一般的には望ましい挙動だが、この例のようにシンプ
ルなケースでは単に速度の低下を招いている。

from theano import function, config, shared, sandbox, Out
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x # cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([],
        Out(sandbox.cuda.basic_ops.gpu_from_host(T.exp(x)),
            borrow=True))
print f.maker.fgraph.toposort()
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()
print 'Looping %d times took' % iters, t1 - t0, 'seconds'
print 'Result is', r
print 'Numpy result is', numpy.asarray(r)
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print 'Used the cpu'
else:
    print 'Used the gpu'
