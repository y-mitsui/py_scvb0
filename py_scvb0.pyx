import numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "lda_scvb0.h":
    ctypedef struct Scvb0:
        double *Theta
        double *Phi
        
    Scvb0* scvb0Init(int n_topic, int n_iter, int batch_size, double alpha, double beta)
    void scvb0Fit(Scvb0 *ctx, int** word_indexes_ptr, unsigned short** word_counts_ptr, int* n_word_each_doc, int* n_word_type_each_doc,unsigned long n_all_word, int n_document, int n_word_type)
    
cdef class PyScvb0:
    cdef Scvb0* scvb0
    cdef int n_topics
    
    def __init__(self, n_topics, alpha, beta, n_iter=2000, batch_size=256):
        self.scvb0 = scvb0Init(n_topics, n_iter, batch_size, alpha, beta)
        self.n_topics = n_topics
    
    def __dealloc__(self):
        free(self.scvb0.Theta)
        free(self.scvb0.Phi)
        free(self.scvb0)
    
    def fit(self, corpus):
        cdef int n_document = len(corpus)
        cdef int** word_indexes_ptr = <int **>malloc(sizeof(int*) * n_document)
        cdef unsigned short** word_counts_ptr = <unsigned short **>malloc(sizeof(unsigned short*) * n_document)
        cdef int n_word_type = 0
        cdef int n_all_word = 0
        cdef int *n_word_type_each_doc = <int *>malloc(sizeof(int) * n_document)
        cdef int *n_word_each_doc = <int *>malloc(sizeof(int) * n_document)
        
        for i, row in enumerate(corpus):
            n_word_type_each_doc[i] = len(row)
            word_indexes_ptr[i] = <int*>malloc(sizeof(int) * n_word_type_each_doc[i])
            word_counts_ptr[i] = <unsigned short*>malloc(sizeof(unsigned short) * n_word_type_each_doc[i])
            n_word_each_doc[i] = 0
            for j, (w_i, w_c) in enumerate(row):
                word_indexes_ptr[i][j] = w_i
                word_counts_ptr[i][j] = w_c
                n_word_each_doc[i] += word_counts_ptr[i][j]
                if n_word_type < w_i:
                    n_word_type = w_i
            n_all_word += n_word_each_doc[i]
        n_word_type += 1
            
        scvb0Fit(self.scvb0, word_indexes_ptr, word_counts_ptr, n_word_each_doc, n_word_type_each_doc, n_all_word, n_document, n_word_type)
        theta = []
        print("a")
        for i in range(n_document):
            row_theta = []
            for j in range(self.n_topics):
                row_theta.append(self.scvb0.Theta[i * self.n_topics + j])
            theta.append(row_theta)
        print("b")
        phi = []
        for i in range(n_word_type):
            row_phi = []
            for j in range(self.n_topics):
                row_phi.append(self.scvb0.Phi[i * self.n_topics + j])
            phi.append(row_phi)
        
        return np.array(theta), np.array(phi).T
    
    
        
