# py_scvb0
確率的周辺化変分ベイズ法による、潜在的ディリクレ配分法のPythonライブラリ  
このアルゴリズムによって、より省メモリで高速なLDAの学習が可能になります。  
一般的なPCを使った日本語版Wikipedia全記事の学習など、今までの技術では難しかったタスクもこの技術を使うことで遂行できる可能性があります。  
精度の高いニュース記事のレコメンデーションや特許検索などのタスクに有効です。  
C言語で実装後、Pythonラッピングしています。  
# インストール
このライブラリはcython、gensim、numpyを使っていますので、まだの方はインストールをお願いします。  
Ubuntu20.04の場合のインストール手順
```
$ sudo python3 setup.py install 
```
# 使い方
```
class py_scvb0.PyScvb0(n_topics=10, alpha=0.1, beta=0.01, n_iter=2000, batch_size=256, n_thread=1)
n_topics: int: トピック数
alpha: float: theta(文書パラメーター)に対する正則化.小さい値ほど強い（スパースになりやすい）
alpha: float: phi(単語パラメーター)に対する正則化.小さい値ほど強い
n_iter: int: 学習イテレーション回数
batch_size: int: ミニバッチで使う文書数
```
メソッド：
```
fit(corpus): corpusを基に学習します。戻り値は(文書ごとのトピック分布、単語ベクトル)のタプル
transformSingle(self, corpus_row): 学習結果を基に、新しく観測された一つの文書のトピック分布を獲得します。
```
corpusのフォーマットはgensimと同じになります。  
より詳しい説明はexample.pyをご覧ください。  
文書分布は２次元配列で、文書の数だけトピック分布が格納されています。  
文書間の類似度は文書分布をKLタイバージェンスで計測します。  
また、文書分布と単語ベクトルの内積をとった行列を使うことでクロスバリデーション尤度が計算できますので、PyScvb0で指示するハイパーパラメーターに客観性を与えることが出来ます。
