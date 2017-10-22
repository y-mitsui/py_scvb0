from py_scvb0 import PyScvb0
from gensim.matutils import corpus2dense
import numpy as np

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
theta, phi = py_scvb0.fit(corpus)

mat_A = corpus2dense(corpus, phi.shape[1]).T
print("original:")
print(mat_A / np.sum(mat_A, 1).reshape(-1, 1))
print("reverse:")
print(np.dot(theta, phi))

py_scvb0 = PyScvb0(3, 3., 3., n_iter=1000, batch_size=len(corpus))
theta, phi = py_scvb0.fit(corpus)
