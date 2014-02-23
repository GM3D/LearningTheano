TheanoからGPUが使えているかどうかのチェック

以下のスクリプトを用意する。処理の内容は単に多くの乱数を生成し、その
exp()を計算しているのみである。入力xがGPU上に確保されるようにshared変
数を使用している。

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print f.maker.fgraph.toposort()
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()
print 'Looping %d times took' % iters, t1 - t0, 'seconds'
print 'Result is', r
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print 'Used the cpu'
else:
    print 'Used the gpu'

これを仮にgpu-test.pyというファイル名でセーブしたとする。以下の二通り
のコマンドラインでこのスクリプトを実行してみると、後者では正しくTheano
からCUDAが使用できる設定になっていれば、"Used the gpu"と表示されるはず
である。

$ THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python check1.py
[Elemwise{exp,no_inplace}(<TensorType(float32, vector)>)]
Looping 1000 times took 3.06635117531 seconds
Result is [ 1.23178029  1.61879337  1.52278066 ...,  2.20771813  2.29967761
  1.62323284]
Used the cpu

$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python check1.py
Using gpu device 0: GeForce GTX 580
[GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>), HostFromGpu(GpuElemwise{exp,no_inplace}.0)]
Looping 1000 times took 0.638810873032 seconds
Result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
  1.62323296]
Used the gpu

デバイス上に割り当てられたデータへのハンドル
上の例では、関数の値を返すのに、デバイスメモリからホストメモリにコピー
されたNumpy ndarrayを用いていたので、速度の向上は限定的だった。
一方でそのためにdevice=gpuを指定しただけで速くなったのではあるが、もし
ポータビリティが下がるのを気にしなければ、計算グラフをGPU上に格納され
た結果を用いるように構成することで、より大きなスピードアップが得られる。
gpu_from_host演算子は入力をホストからGPUへコピーする操作を表しているが、
上の例でT.exp()をGPU版に置き換えると、最適化によってこの操作が取り除か
れる。

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

上のスクリプトの実行例:

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python check2.py
Using gpu device 0: GeForce GTX 580
[GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>)]
Looping 1000 times took 0.34898686409 seconds
Result is <CudaNdarray object at 0x6a7a5f0>
Numpy result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
  1.62323296]
Used the gpu

このように、結果の配列をホスト側にコピーして返すのをやめるだけで、約
50%の実行時間を短縮できる。各関数呼び出しで返されるオブジェクトは、
Numpy ndarrayではなく、CudaNdarrayとなる。これは必要に応じて通常の
Numpyのキャストを用いてNumpy ndarrayに変換できる。

GPUをフルスピードで動かすには
このシンプルな例で最大限のパフォーマンスを得るには、Theanoに返り値のコ
ピーを行わせないために、borrow=Trueフラグを設定してoutインスタンスを使
う必要がある。これは、Theanoは内部での使用のためにメモリを(ワーキング
バッファとして)あらかじめ確保するが、ここに得られた結果を直接ユーザに
返すことはないためである。その代わりにTheanoは関数呼び出し毎に新しくメ
モリを割り当て、そこにこのバッファの内容をコピーして返す。これはその後
の関数呼び出しによって、以前に計算した結果を上書きしてしまうことがない
ようにするためである。通常はこれが望ましい動作であるが、この例のような
シンプルなケースでは、これは単に不必要なコピーによって実効速度を低下さ
せているだけである。

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

このバージョンのコードはわずか0.05秒あまりで実行され、CPUによる実装の
およそ60倍のスピードアップとなる。

borrowフラグがFalseの場合:

$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python using_gpu_solution_1.py
Using gpu device 0: GeForce GTX 580
[GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>)]
Looping 1000 times took 0.31614613533 seconds
Result is <CudaNdarray object at 0x77e9270>
Numpy result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
  1.62323296]
Used the gpu

borrowフラグがTrueの場合

$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python using_gpu_solution_1.py
Using gpu device 0: GeForce GTX 580
[GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>)]
Looping 1000 times took 0.0502779483795 seconds
Result is <CudaNdarray object at 0x83e5cb0>
Numpy result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
  1.62323296]
Used the gpu

このborrowフラグをTrueに設定した例は若干危険である。というのは、rにいっ
たん関数呼び出しの返り値を保存したら、その後の関数呼び出しで値が上書き
されてしまう可能性があることを計算にいれておかなければならないからであ
る。

この例ではborrow=Trueを設定することで劇的な性能の改善が得られるが、そ
の効果はもっと大きな計算グラフではずっと弱くなり、一方で起こりうるメモ
リの上書きを正しく扱い損ねる危険はずっと大きくなることに注意して欲しい。

GPUで高速化できる演算
Theanoのパフォーマンス上の特性は実装が最適化されていくにつれて変わって
いくであろうし、またデバイスによっても異なってくる。しかし今現在想定で
きる事項にはつぎのようなものがある:

高速化できるのはfloat32による計算のみであり、float64のサポートの改善は
次世代のハードウェアを待つことになり、またそれが実現しても比較的遅いだ
ろう。(2010年1月時点)

行列の積、畳み込み、大きいサイズの成分毎の演算は、引数が30個のプロセッ
サを使い切るほど十分大きければ大幅に高速化される。(5-50倍)

添字操作、ディメンションシャッフル、定数時間の行列のシェイプ変更はGPU
でもCPUと同等の速度になる。

行や列に沿ってのテンソルの和は、GPUではCPUより若干遅くなる。

大量のデータをデバイスとホスト間でコピーする操作は比較的低速であり、し
ばしば一つや二つの関数をGPU化するメリットを帳消しにしてしまう。GPUのパ
フォーマンスを生かすためには、データの転送コストに見合うだけの作業をさ
せることである。

GPUのパフォーマンスを向上させる秘訣
GPUを多く使用するつもりならば、.theanorcにfloatX=float32を加えるとよい
かもしれない。

Theanoのallow_gc=Falseフラグを使用する。(「GPUの非同期機能」を参照)

floatX=float32が設定されている場合、matrix、vector、scalarコンストラク
タはこの設定に従うので、dmatrix、dvector、dscalarよりもこれらを使用す
る。

出力用の変数にfloat64ではなくfloat32型を用いること。グラフ中でfloat32
を多用するほどGPUで計算できる部分が増える。

GPUデバイスへのデータ転送を最小化するため、頻繁にアクセスされる変数に
はshared変数を用いること。GPUを使用する場合、float32型のテンソルは演算
に伴う転送時間をなくすため、デフォルトでGPU上に置かれる。

関数のパフォーマンスに問題がある場合、mode='ProfileMode'フラグを追加し
て関数をコンパイルしてみるとよい。するとプログラムの終了時に実行時間に
ついての情報を出力してくれる。これを見て実行時間が正常かどうか確認する。
もしある特定の演算子やApplyノードが時間を取りすぎているようなら、GPUプ
ログラミングの知識があるならtheano.sandbox.cudaの中を見てみるとよい。
gpu演算子のSpent Xs(X%) in cpu op, Xs(X%)の行や、転送演算子のXs(X%)の
部分を見てみると、グラフ中の演算が十分にGPU上で実行されていなかったり、
メモリの転送が多すぎたりといったケースをチェックできる。

nvccのオプションを活用する。nvccには計算を高速化するためのいくつかのオ
プションがある。-ftz=trueは正規化できない浮動小数を0に
し、-prec-div=falseと-prec-sqrt=falseはそれぞれ除算と平方根の精度を犠
牲にして速度を向上させる。Theanoフラグでnvcc.flags=-use_fast_mathを設
定することによってこれらをすべて有効にできる。あるいは、
nvcc.flags=-ftz=tru -prec-div=falseのように個別に有効、無効を設定でき
る。

GPUの非同期機能
Theano 0.6以降、GPUの非同期機能が利用できるようになりつつある。
