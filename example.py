from py_scvb0 import PyScvb0
from gensim.matutils import corpus2dense

corpus = [
            [(0, 1), (1, 2), (2, 3)],
            [(0, 1), (1, 2), (2, 3)],
            [(0, 1), (1, 2), (2, 3)],
      ]
py_scvb0 = PyScvb0(10, 1., 0.1)
theta, phi = py_scvb0.fit(corpus)

mat_A = corpus2dense(corpus, phi.shape[1]).T
print(mat_A / np.sum(mat_A, 1).reshape(-1, 1))
print(np.dot(theta, phi))
