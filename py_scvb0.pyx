import numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "lda_scvb0.h":
    ctypedef struct Scvb0:
        double *Theta
        double *Phi
        int n_topic
        int n_document
        int n_word_type
        
    Scvb0* scvb0Init(int n_topic, int n_iter, int batch_size, int n_thread, double alpha, double beta)
    void scvb0Fit(Scvb0 *ctx, int** word_indexes_ptr, unsigned short** word_counts_ptr, int* n_word_each_doc, int* n_word_type_each_doc,unsigned long n_all_word, int n_document, int n_word_type)
    void scvb0Save(Scvb0 *ctx, const char *path)
    Scvb0 *scvb0Load(const char *path)
    Scvb0 *scvb0FitFile(Scvb0 *ctx, const char *path)
    double *scvb0TransformSingle(Scvb0 *ctx, int *doc_word, int n_word, int max_iter)
    int scvb0Sample(const char *path, int ***word_indexes_r, unsigned short ***word_counts_r, int **n_word_type_each_doc_r, int **n_word_each_doc_r, unsigned long long *n_all_word_r, int *n_document_r, int *n_word_type_r)
    
cdef class PyScvb0:
    cdef Scvb0* scvb0
    cdef int n_topics
    
    def __init__(self, n_topics=10, alpha=0.1, beta=0.01, n_iter=2000, batch_size=256, n_thread=1):
        self.scvb0 = scvb0Init(n_topics, n_iter, batch_size, n_thread, alpha, beta)
        self.n_topics = n_topics
    
    def save(self, file_path):
        scvb0Save(self.scvb0, file_path)
        
    def load(self, file_path):
        self.scvb0 = scvb0Load(file_path)
        self.n_topics = self.scvb0.n_topic
    
    def __dealloc__(self):
        free(self.scvb0.Theta)
        free(self.scvb0.Phi)
        free(self.scvb0)
    
    def transformSingle(self, corpus_row):
        n_words = np.sum(map(lambda x:x[1], corpus_row))
        cdef int *doc_word = <int*>malloc(sizeof(int) * n_words)
        n_words = 0
        for w_i, w_c in corpus_row:
            for i in range(w_c):
                doc_word[n_words] = w_i
                n_words += 1
                
        cdef double *result = scvb0TransformSingle(self.scvb0, doc_word, n_words, 20)
        topics = []
        for i in range(self.n_topics):
            topics.append(result[i])
        return np.array(topics) / np.sum(topics)
    
    def _getParams(self):
        theta = []
        for i in range(self.scvb0.n_document):
            row_theta = []
            for j in range(self.n_topics):
                row_theta.append(self.scvb0.Theta[i * self.n_topics + j])
            theta.append(row_theta)
            
        phi = []
        for i in range(self.scvb0.n_word_type):
            row_phi = []
            for j in range(self.n_topics):
                row_phi.append(self.scvb0.Phi[i * self.n_topics + j])
            phi.append(row_phi)
        
        return np.array(theta), np.array(phi).T
        
    def getSample(self, path):
        cdef int **word_indexes
        cdef unsigned short **word_counts
        cdef int *n_word_type_each_doc, *n_word_each_doc
        cdef int n_document, n_word_type
        cdef unsigned long long n_all_word
        
        scvb0Sample(path, &word_indexes, &word_counts, &n_word_type_each_doc, &n_word_each_doc, &n_all_word, &n_document, &n_word_type);
        corpus = []
        for i in range(n_document):
            row = []
            for j in range(n_word_type_each_doc[i]):
                row.append((word_indexes[i][j], word_counts[i][j]))
            corpus.append(row)
        
        free(word_indexes);
        free(word_counts);
        free(n_word_type_each_doc);
        free(n_word_each_doc);
        return corpus
        
    def fitFile(self, path):
        scvb0FitFile(self.scvb0, path)
        return self._getParams()
    
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
        return self._getParams()
    
    
        
