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

この例では、0.05秒強で実行が終了し、CPUでの実装に比べて60倍以上の向上
になっている。

borrowフラグをFalseに設定:

$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python using_gpu_solution_1.py
Using gpu device 0: GeForce GTX 580
[GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>)]
Looping 1000 times took 0.31614613533 seconds
Result is <CudaNdarray object at 0x77e9270>
Numpy result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
  1.62323296]
Used the gpu

borrowフラグをTrueに設定:

$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python using_gpu_solution_1.py
Using gpu device 0: GeForce GTX 580
[GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>)]
Looping 1000 times took 0.0502779483795 seconds
Result is <CudaNdarray object at 0x83e5cb0>
Numpy result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
  1.62323296]
Used the gpu

