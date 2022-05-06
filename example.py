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
py_scvb0 = PyScvb0(3, 3., 3., n_iter=1000, batch_size=len(corpus))
#メモリーリークが発生するので同じインスタンスでfitを何度も呼び出さない。
theta, phi = py_scvb0.fit(corpus)
#　疎行列表現から密行列表現に変換します。mat_Aの一次元目は文書、２次元目は単語、mat_A[1][2]は２つ目の文書の単語ID２が幾つ入っているかが示されます
mat_A = corpus2dense(corpus, phi.shape[1]).T
print("original:")
#行に対して足して１になるように確率正規化します。
print(mat_A / np.sum(mat_A, 1).reshape(-1, 1))
#ここのreverse出力によって規則性が正しく捉えられているか確認します。最後の文書が他の文書と同じになっていれば成功です。
print("reverse:")
print(np.dot(theta, phi))

