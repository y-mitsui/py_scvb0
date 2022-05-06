from py_scvb0 import PyScvb0
from gensim.matutils import corpus2dense
import numpy as np

# corpusの一次元目は文書、２次元目は単語IDとその単語が文書中に含まれる数
# この場合、corpus[0][2]は１番目の文書にID2を持つ単語が３つ含まれていることが分かります
# 最初の６つの文書で規則性を作り、最後の文書でID2の単語を欠損状態にして、欠損単語が規則性に従い、正しく3と推定できるかを実験します。
corpus = [
            [(0, 1), (1, 2), (2, 3)],
            [(0, 1), (1, 2), (2, 3)],
            [(0, 1), (1, 2), (2, 3)],
            [(0, 1), (1, 2), (2, 3)],
            [(0, 1), (1, 2), (2, 3)],
            [(0, 1), (1, 2), (2, 3)],
            [(0, 1), (1, 2)],
      ]
py_scvb0 = PyScvb0(3, 1., 1., n_iter=1000, batch_size=len(corpus))
#メモリーリークが発生するので同じインスタンスでfitを何度も呼び出さない。
theta, phi = py_scvb0.fit(corpus)
#　疎行列表現から密行列表現に変換します。mat_Aの一次元目は文書、２次元目は単語、mat_A[1][2]は２つ目の文書の単語ID２が幾つ入っているかが示されます
mat_A = corpus2dense(corpus, phi.shape[1]).T
print("original:")
#行に対して足して１になるように確率正規化します。
print(mat_A / np.sum(mat_A, 1).reshape(-1, 1))
#ここのestimate出力によって規則性が正しく捉えられているか確認します。最後の文書が他の文書と概ね同じになっていれば成功です。
#この出力結果の解釈としては、文書から単語が生成される真の（すなわちスムージングされた）確率の推定値を表します。
#また、このestimateで出力される行列を使うとクロスバリデーション尤度が計算できますので、PyScvb0で指示するハイパーパラメーターに客観性を与えることが出来ます。
print("estimate:")
print(np.dot(theta, phi))

#学習結果を基に新しく観測された１つの文書のトピック分布を得ます
topic_dist = py_scvb0.transformSingle([(0, 1), (1, 2), (2, 3)])
#ここの出力がestimateで出力された他の文書の内容とほぼ同じになっていれば成功です
print(np.dot([topic_dist], phi))

