# coding: UTF-8
# TheanoからGPUが使えているかどうかのチェック

# 以下のスクリプトを用意する。処理の内容は単に多くの乱数を生成し、その
# exp()を計算しているのみである。入力xがGPU上に確保されるようにshared変
# 数を使用している。

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

# これを仮にgpu-test.pyというファイル名でセーブしたとする。以下の二通り
# のコマンドラインでこのスクリプトを実行してみると、後者では正しくTheano
# からCUDAが使用できる設定になっていれば、"Used the gpu"と表示されるはず
# である。

# $ THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python check1.py
# [Elemwise{exp,no_inplace}(<TensorType(float32, vector)>)]
# Looping 1000 times took 3.06635117531 seconds
# Result is [ 1.23178029  1.61879337  1.52278066 ...,  2.20771813  2.29967761
#   1.62323284]
# Used the cpu

# $ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python check1.py
# Using gpu device 0: GeForce GTX 580
# [GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>), HostFromGpu(GpuElemwise{exp,no_inplace}.0)]
# Looping 1000 times took 0.638810873032 seconds
# Result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
#   1.62323296]
# Used the gpu

# デバイス上に割り当てられたデータへのハンドル
# 上の例では、関数の値を返すのに、デバイスメモリからホストメモリにコピー
# されたNumpy ndarrayを用いていたので、速度の向上は限定的だった。
# 一方でそのためにdevice=gpuを指定しただけで速くなったのではあるが、もし
# ポータビリティが下がるのを気にしなければ、計算グラフをGPU上に格納され
# た結果を用いるように構成することで、より大きなスピードアップが得られる。
# gpu_from_host演算子は入力をホストからGPUへコピーする操作を表しているが、
# 上の例でT.exp()をGPU版に置き換えると、最適化によってこの操作が取り除か
# れる。
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

# 上のスクリプトの実行例:

# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python check2.py
# Using gpu device 0: GeForce GTX 580
# [GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>)]
# Looping 1000 times took 0.34898686409 seconds
# Result is <CudaNdarray object at 0x6a7a5f0>
# Numpy result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
#   1.62323296]
# Used the gpu

# このように、結果の配列をホスト側にコピーして返すのをやめるだけで、約
# 50%の実行時間を短縮できる。各関数呼び出しで返されるオブジェクトは、
# Numpy ndarrayではなく、CudaNdarrayとなる。これは必要に応じて通常の
# Numpyのキャストを用いてNumpy ndarrayに変換できる。

# GPUをフルスピードで動かすには
# このシンプルな例で最大限のパフォーマンスを得るには、Theanoに返り値のコ
# ピーを行わせないために、borrow=Trueフラグを設定してoutインスタンスを使
# う必要がある。これは、Theanoは内部での使用のためにメモリを(ワーキング
# バッファとして)あらかじめ確保するが、ここに得られた結果を直接ユーザに
# 返すことはないためである。その代わりにTheanoは関数呼び出し毎に新しくメ
# モリを割り当て、そこにこのバッファの内容をコピーして返す。これはその後
# の関数呼び出しによって、以前に計算した結果を上書きしてしまうことがない
# ようにするためである。通常はこれが望ましい動作であるが、この例のような
# シンプルなケースでは、これは単に不必要なコピーによって実効速度を低下さ
# せているだけである。

from theano import Out

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

# このバージョンのコードはわずか0.05秒あまりで実行され、CPUによる実装の
# およそ60倍のスピードアップとなる。

# borrowフラグがFalseの場合:

# $ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python using_gpu_solution_1.py
# Using gpu device 0: GeForce GTX 580
# [GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>)]
# Looping 1000 times took 0.31614613533 seconds
# Result is <CudaNdarray object at 0x77e9270>
# Numpy result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
#   1.62323296]
# Used the gpu

# borrowフラグがTrueの場合

# $ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python using_gpu_solution_1.py
# Using gpu device 0: GeForce GTX 580
# [GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>)]
# Looping 1000 times took 0.0502779483795 seconds
# Result is <CudaNdarray object at 0x83e5cb0>
# Numpy result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
#   1.62323296]
# Used the gpu

# このborrowフラグをTrueに設定した例は若干危険である。というのは、rにいっ
# たん関数呼び出しの返り値を保存したら、その後の関数呼び出しで値が上書き
# されてしまう可能性があることを計算にいれておかなければならないからであ
# る。

# この例ではborrow=Trueを設定することで劇的な性能の改善が得られるが、そ
# の効果はもっと大きな計算グラフではずっと弱くなり、一方で起こりうるメモ
# リの上書きを正しく扱い損ねる危険はずっと大きくなることに注意して欲しい。

# GPUで高速化できる演算
# Theanoのパフォーマンス上の特性は実装が最適化されていくにつれて変わって
# いくであろうし、またデバイスによっても異なってくる。しかし今現在想定で
# きる事項にはつぎのようなものがある:

# 高速化できるのはfloat32による計算のみであり、float64のサポートの改善は
# 次世代のハードウェアを待つことになり、またそれが実現しても比較的遅いだ
# ろう。(2010年1月時点)

# 行列の積、畳み込み、大きいサイズの成分毎の演算は、引数が30個のプロセッ
# サを使い切るほど十分大きければ大幅に高速化される。(5-50倍)

# 添字操作、ディメンションシャッフル、定数時間の行列のシェイプ変更はGPU
# でもCPUと同等の速度になる。

# 行や列に沿ってのテンソルの和は、GPUではCPUより若干遅くなる。

# 大量のデータをデバイスとホスト間でコピーする操作は比較的低速であり、し
# ばしば一つや二つの関数をGPU化するメリットを帳消しにしてしまう。GPUのパ
# フォーマンスを生かすためには、データの転送コストに見合うだけの作業をさ
# せることである。

# GPUのパフォーマンスを向上させる秘訣
# GPUを多く使用するつもりならば、.theanorcにfloatX=float32を加えるとよい
# かもしれない。

# Theanoのallow_gc=Falseフラグを使用する。(「GPUの非同期機能」を参照)

# floatX=float32が設定されている場合、matrix、vector、scalarコンストラク
# タはこの設定に従うので、dmatrix、dvector、dscalarよりもこれらを使用す
# る。

# 出力用の変数にfloat64ではなくfloat32型を用いること。グラフ中でfloat32
# を多用するほどGPUで計算できる部分が増える。

# GPUデバイスへのデータ転送を最小化するため、頻繁にアクセスされる変数に
# はshared変数を用いること。GPUを使用する場合、float32型のテンソルは演算
# に伴う転送時間をなくすため、デフォルトでGPU上に置かれる。

# 関数のパフォーマンスに問題がある場合、mode='ProfileMode'フラグを追加し
# て関数をコンパイルしてみるとよい。するとプログラムの終了時に実行時間に
# ついての情報を出力してくれる。これを見て実行時間が正常かどうか確認する。
# もしある特定の演算子やApplyノードが時間を取りすぎているようなら、GPUプ
# ログラミングの知識があるならtheano.sandbox.cudaの中を見てみるとよい。
# gpu演算子のSpent Xs(X%) in cpu op, Xs(X%)の行や、転送演算子のXs(X%)の
# 部分を見てみると、グラフ中の演算が十分にGPU上で実行されていなかったり、
# メモリの転送が多すぎたりといったケースをチェックできる。

# nvccのオプションを活用する。nvccには計算を高速化するためのいくつかのオ
# プションがある。-ftz=trueは正規化できない浮動小数を0に
# し、-prec-div=falseと-prec-sqrt=falseはそれぞれ除算と平方根の精度を犠
# 牲にして速度を向上させる。Theanoフラグでnvcc.flags=-use_fast_mathを設
# 定することによってこれらをすべて有効にできる。あるいは、
# nvcc.flags=-ftz=tru -prec-div=falseのように個別に有効、無効を設定でき
# る。

# GPUの非同期機能
# Theano 0.6以降、GPUの非同期機能が利用できるようになりつつある。これに
# よって速度の向上が得られるが、エラーが起きた場合、それが発生した時点で
# すぐに表示されずに、遅れて表示される可能性がある。このためTheanoのApply
# ノードをプロファイリングするのが難しくなる可能性があるが、このような場
# 合は。環境変数CUDA_LAUNCH_BLOCKING=1を設定するとすべてのカーネル呼び出
# しが自動的に同期呼び出しになる。
# これによってパフォーマンスは低下するが、プロファイルデータは正しく得られ、エ
# ラーメッセージのタイミングも望ましいものになる。

# また、非同期機能はTheanoが中間結果に対して行うガベージコレクションと競
# 合する。Theanoはガベージコレクションのためにグラフ中に同期ポイントを挿
# 入するので、非同期機能を十分に生かすためにはガベージコレクションをオフ
# にする必要がある。allow gc=Falseフラグを設定することでさらに速度の向上
# が見込めるが、一方でメモリの使用量は増大する。

# Shared変数の値を変更する
# Shared変数の値を変更するには、すなわちGPUに対して新しいデータを与える
# には、shared変数.set_value(新しい値)を用いる。もっと詳細については、
# 「スピードと正しさのためにメモリのエイリアスについて理解する」を参照。

# 練習問題
# 再び、次のロジスティック回帰を考える。

import theano
rng = numpy.random

N = 400
feats = 784
D = (rng.randn(N, feats).astype(theano.config.floatX),
rng.randint(size=N,low=0, high=2).astype(theano.config.floatX))
training_steps = 10000

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(feats).astype(theano.config.floatX), name="w")
b = theano.shared(numpy.asarray(0., dtype=theano.config.floatX), name="b")
x.tag.test_value = D[0]
y.tag.test_value = D[1]
#print "Initial model:"
#print w.get_value(), b.get_value()

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w)-b)) # Probability of having a one
prediction = p_1 > 0.5 # The prediction that is done: 0 or 1
xent = -y*T.log(p_1) - (1-y)*T.log(1-p_1) # Cross-entropy
cost = xent.mean() + 0.01*(w**2).sum() # The cost to optimize
gw,gb = T.grad(cost, [w,b])

# Compile expressions to functions
train = theano.function(
            inputs=[x,y],
            outputs=[prediction, xent],
            updates={w:w-0.01*gw, b:b-0.01*gb},
            name = "train")
predict = theano.function(inputs=[x], outputs=prediction,
            name = "predict")

if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for x in
        train.maker.fgraph.toposort()]):
    print 'Used the cpu'
elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
          train.maker.fgraph.toposort()]):
    print 'Used the gpu'
else:
    print 'ERROR, not able to tell if theano used the cpu or the gpu'
    print train.maker.fgraph.toposort()

for i in range(training_steps):
    pred, err = train(D[0], D[1])
#print "Final model:"
#print w.get_value(), b.get_value()

print "target values for D"
print D[1]

print "prediction on D"
print predict(D[0])

# この例を、floatX=gloat32に設定して、timeコマンドを用いてコマンドライン
# 上で実行時間を測定してみてほしい。「設定とコンパイルモード」の項での
# 解答を用いてもかまわない

# GPUを用いて、CPUに対して速度の向上があったか
# それはどこで生じたものか(ProfileModeを使って調べる)
# さらに速度を向上させるにはどのような方法があるか考え、実際にテストしてみよ。

# 注意:
# 現在サポートされているのは32ビットのfloat型のみである(他は開発中)
# float32のshared型変数は、デフォルトでGPU上に移される
# 1プロセスが利用できるのは1つのGPUに限られる。
# GPUデバイスを使用するよう要求するには、device=gpuフラグを使用する。
# 複数のGPUがある場合は、device=gpu{0, 1, ...}を使用する。
# TheanoフラグfloatX=float32をコード中で(theano.config.floatXで)指定する。
# shared変数に代入する前に、必要なキャストを行う。
# 次のような方法で、int32やfloat32がfloat64にキャストされないようにする。
# 明示的なキャストをコードで行うか、[u]int[8, 16]を用いる。
#平均オペレータの前後にも手動でキャストを挿入する(配列の長さによる除算
#がint64を含んでいる)。
# なお、現在新しいキャストの機構が開発中である。

# GPUを直接プログラムするソフトウェア
# Theanoはメタプログラムツールといってよいが、直接GPUをプログラムするツー
# ルとしては以下のようなものがある。

# CUDA: NVIDIAが提供する、C言語の拡張版(CUDA C)に基づくGPUプログラミン
# グAPI
# ベンダー依存
# 数値演算ライブラリ(BLAS、RNG、FFT)が充実しつつある
# OpenCL: CUDAのマルチベンダー版にあたるもの
#  より一般的で、標準化されている
#  ライブラリはCUDA環境より少なく、利用も少ない
#  PyCUDA: PythonからCUDAのAPIにアクセスするためのバインディング
# 簡便性:
#  PythonからGPUメタプログラミングを容易に行える
#  PythonからCUDAのコードをコンパイルできる抽象化モジュール
#  (pycuda.driver.SourceModule)
#  GPUメモリバッファへのアクセス(pycuda.gpuarray.GPUArray)
#  役に立つドキュメント

# 完全性: CUDAの全APIへのバインディングが提供されている
# 自動的なエラーチェッキング: 全てのCUDAのエラーがPythonの例外に自動的に
# 変換される。
# スピード: pyCUDAのベースレイヤーはC++で記述されている。
# GPUオブジェクトに対する良好なメモリ管理:
#  オブジェクトのクリーンアップはオブジェクトの寿命と同期している(いわ
#  ゆるRAII、Resource Aquisition Is Initialization)
 # メモリリークやクラッシュのない正しいコードを書くのが非常に容易である
 # PyCUDAはオブジェクト間の依存性を管理してくれる(例えば、コンテクスト
 # 内で確保されたメモリがすべて開放されるまでコンテクストをデタッチし
 # ない)

# (以上はPyCUDAのドキュメント及び、PyCUDAに関するAdnreas Kloecknerのウェ
#  ブサイトからの引用である)

# PyCUDAによるプログラミングを学ぶには
# すでにC言語について十分な知識があるのであれば、まずCのCUDA拡張(CUDA C)
# を用いたGPUプログラムを学び、ついでPythonのラッパーを用いてCUDA APIに
# アクセスするためにPyCUDAを学のがよいかもしれない。

# 学習の参考として、以下のリソースをあげておく。

# CUDA APIとCUDA C: 入門編
#  NVIDIAのスライド
#  NY大 Steinのスライド

# CUDA APIとCUDA C: 上級編
# MIT IAP2009 CUDA (全範囲の講義、Kirk-Hwuによるテキスト、例と追加リソー
# ス)
# イリノイ大の講義(全範囲、Kirk-Hwuのテキスト)
#  NVIDIAナレッジベース(入門から高度なものまで広範にわたる)
#  StackOverflowのpractical issues
#  CUDA optimization

# PyCUDA: 入門編
# Kloecknerのスライド
# Kloecknerのウェブサイト

# PyCUDA: 上級編
# PyCUDAドキュメントサイト

# 以下の例は、PyCUDAによるGPUプログラミングのさわりである。自信がついた
# らその後の練習問題を試してみるとよいだろう。

# 例: PyCUDA
# (from PyCUDA's documentation)
import pycuda.autoinit
import pycuda.driver as drv

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

multiply_them = mod.get_function("multiply_them")

a = numpy.random.randn(400).astype(numpy.float32)
b = numpy.random.randn(400).astype(numpy.float32)

dest = numpy.zeros_like(a)
multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(400,1,1), grid=(1,1))

assert numpy.allclose(dest, a*b)
print dest

# 練習問題
# 上の例を実行してみよ。そして(20, 10)行列を扱うように変更してみよ。

# 例: TheanoとPyCUDAの組み合わせ
import theano.sandbox.cuda as cuda

class PyCUDADoubleOp(theano.Op):
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def __str__(self):
        return self.__class__.__name__
    def make_node(self, inp):
        inp = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp))
        assert inp.dtype == "float32"
        return theano.Apply(self, [inp], [inp.type()])
    def make_thunk(self, node, storage_map, _, _2):
        mod = SourceModule("""
    __global__ void my_fct(float * i0, float * o0, int size) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<size){
        o0[i] = i0[i]*2;
    }
  }""")
        pycuda_fct = mod.get_function("my_fct")
        inputs = [ storage_map[v] for v in node.inputs]
        outputs = [ storage_map[v] for v in node.outputs]
        def thunk():
            z = outputs[0]
            if z[0] is None or z[0].shape!=inputs[0][0].shape:
                z[0] = cuda.CudaNdarray.zeros(inputs[0][0].shape)
            grid = (int(numpy.ceil(inputs[0][0].size / 512.)),1)
            pycuda_fct(inputs[0][0], z[0], numpy.intc(inputs[0][0].size),
                       block=(512,1,1), grid=grid)
        return thunk


x = theano.tensor.fmatrix()
f = theano.function([x], PyCUDADoubleOp()(x))
xv=numpy.ones((4,5), dtype="float32")
assert numpy.allclose(f(xv), xv*2)
print numpy.asarray(f(xv))

# 練習問題
# 上の例を実行してみよ。そして以下の変更を加えてみよ。

# 二つの行列の掛け算 x*y を行うようにする。
# x + yと x - y、両方の結果を一度に返すようにする。
# (Theanoの現在のelemwise fusion操作の最適化は、単一の出力の場合にのみ適
#  用されることに注意せよ。したがって、ここでの例以上に効率化したければ、
#  +と-操作をまとめて明示的に最適化されたコードを書かなければならない。)
# strideパラメータをサポートするように変更する。(すなわち、入力がCの配
# 列のように連続して並んでいる場合以外にも対応できるようにする)


