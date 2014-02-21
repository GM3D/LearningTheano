# Theanoの設定とコンパイルモード

configモジュールにはTheanoの動作に影響するいくつかのアトリビュートが含まれている。
これらのうち多くがtheanoモジュールのimport中に参照される。そのうちいくつかは、読み込み専用であることが仮定されている。

原則として、ユーザーのコードからconfigモジュール内のアトリビュートを変更してはならない。

Theanoのコードにはこれらのアトリビュートのデフォルトの値が設定されているが、これらの値を.theanorcファイルによって上書きしたり、THEANO_FLAGS環境変数によって変更したりすることが可能である。

評価の順番は、
1. theano.config.<プロパティ名>での設定
2. THEANO_FLAGS
3. .theanorc(もしくは環境変数THEANORCで設定したファイル)での設定

となる。
現在有効な設定は、いつでもtheano.configを表示させることでチェックできる。たとえばすべてのアクティブな設定変数の一覧を見るには、コマンドラインから

python -c 'import theano; print theano.config' | less

とすればよい。
さらに詳しくは、Library/Configurationを参照。

練習問題
次のロジスティック回帰を考える。(ファイルconfig-lr0.py)


# この例を、CPU上(デフォルト)で、floatX=float32で実行されるように変更し(config-lr1.py)、timeコマンドで実行速度を測定せよ。

# 解答: config-lr1.py

# 変更点:
# > theano.config.floatX = 'float32'
# < cost = xent.mean() + 0.01 * (w ** 2).sum()
# > cost = T.cast(xent.mean(), 'float32') + 0.01 * (w ** 2).sum()

