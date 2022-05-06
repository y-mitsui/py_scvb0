#ifndef __LDA_SCVB0_H__
#define __LDA_SCVB0_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n_topic;
    int n_iter;    
    int n_document;
    int batch_size;
    int n_thread;
    int n_word_type;
    unsigned long long n_all_word;
    
    double alpha;
    double beta;
    
    double rhoPhi;
    double rhoTheta;
    
    double *nTheta;
    double *nPhi;
    double *nz;
    double *Phi;
    double *Theta;
    
    double *gamma;
    double *nzHat;
    double *nPhiHat;
    
}Scvb0;

typedef struct {
	int thread_no;
	int *doc_indxes;
	Scvb0 *ctx;
	int n_document;
	int** word_indexes_ptr;
	unsigned short** word_counts_ptr;
	int* n_word_each_doc;
	int* n_word_type_each_doc;
}ThreadArgs;


Scvb0* scvb0Init(int n_topic, int n_iter, int batch_size, int n_thread, double alpha, double beta);
void scvb0Fit(Scvb0 *ctx, int** word_indexes_ptr, unsigned short** word_counts_ptr, int* n_word_each_doc, int* n_word_type_each_doc,unsigned long long n_all_word, int n_document, int n_word_type);
void scvb0Save(Scvb0 *ctx, const char *path);
Scvb0 *scvb0Load(const char *path);
void scvb0EstPhi(Scvb0 *ctx, double *Phi);
double *scvb0TransformSingle(Scvb0 *ctx, int *doc_word, int n_word, int max_iter);
double *scvb0FitTransform(Scvb0 *ctx, int** word_indexes_ptr, unsigned short** word_counts_ptr, int* n_word_each_doc, int* n_word_type_each_doc,unsigned long long n_all_word, int n_document, int n_word_type);
void scvb0Free(Scvb0* ctx);

int scvb0Sample(const char *path, int ***word_indexes_r, unsigned short ***word_counts_r, int **n_word_type_each_doc_r, int **n_word_each_doc_r, unsigned long long *n_all_word_r, int *n_document_r, int *n_word_type_r);
void scvb0FitFile(Scvb0 *ctx, const char *path);

#ifdef __cplusplus
}
#endif

#endif /* __CHASEN_H__ */
