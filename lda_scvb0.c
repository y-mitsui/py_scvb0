#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include "lda_scvb0.h"

#define d_malloc(type, numb) (type*)malloc(sizeof(type) * numb)
#define d_calloc(type, numb) (type*)calloc(sizeof(type), numb)

unsigned long xor128(){ 
    static unsigned long x=123456789,y=362436069,z=521288629,w=88675123; 
    unsigned long t; 
    t=(x^(x<<11));x=y;y=z;z=w; return( w=(w^(w>>19))^(t^(t>>8)) ); 
} 

void scvb0Save(Scvb0 *ctx, const char *path){
    FILE *fp;
    
    if (!(fp=fopen(path,"wb"))) {
        perror(path);
        return;
    }
    fwrite(&ctx->n_topic, sizeof(int), 1, fp);
    fwrite(&ctx->n_iter, sizeof(int), 1, fp);
    fwrite(&ctx->n_document, sizeof(int), 1, fp);
    fwrite(&ctx->batch_size, sizeof(int), 1, fp);
    fwrite(&ctx->n_word_type, sizeof(int), 1, fp);
    fwrite(&ctx->n_all_word, sizeof(unsigned long long), 1, fp);
    fwrite(&ctx->alpha, sizeof(double), 1, fp);
    fwrite(&ctx->beta, sizeof(double), 1, fp);
    
    //fwrite(ctx->nTheta, sizeof(double), ctx->n_document * ctx->n_topic, fp);
    fwrite(ctx->Phi, sizeof(double), ctx->n_word_type * ctx->n_topic, fp);
    //fwrite(ctx->nz, sizeof(double), ctx->n_topic, fp);
    fclose(fp);
    
}

Scvb0 *scvb0Load(const char *path){
    FILE *fp;
    Scvb0 *ctx = d_malloc(Scvb0, 1);
    
    if(!(fp = fopen(path/*"data/scvb_data.dat"*/,"rb"))) {
        return NULL;
    }
    fread(&ctx->n_topic, sizeof(int), 1, fp);
    fread(&ctx->n_iter, sizeof(int), 1, fp);
    fread(&ctx->n_document, sizeof(int), 1, fp);
    fread(&ctx->batch_size, sizeof(int), 1, fp);
    fread(&ctx->n_word_type, sizeof(int), 1, fp);
    fread(&ctx->n_all_word, sizeof(unsigned long long), 1, fp);
    fread(&ctx->alpha, sizeof(double), 1, fp);
    fread(&ctx->beta, sizeof(double), 1, fp);
    
    //ctx->gamma = d_malloc(double, ctx->n_topic);
    //ctx->nzHat = d_malloc(double, ctx->n_topic);
    //ctx->nPhiHat = d_malloc(double, ctx->n_word_type * ctx->n_topic);
    //ctx->nz = d_malloc(double, ctx->n_topic);
    //ctx->nPhi = d_malloc(double, ctx->n_word_type * ctx->n_topic);
    //ctx->nTheta = malloc(sizeof(double) * ctx->n_document * ctx->n_topic);
    ctx->Phi = d_malloc(double, ctx->n_word_type * ctx->n_topic);
    //ctx->Theta = malloc(sizeof(double) * ctx->n_document * ctx->n_topic);
    
    //fwrite(ctx->nTheta, sizeof(double), ctx->n_document * ctx->n_topic, fp);
    fread(ctx->Phi, sizeof(double), ctx->n_word_type * ctx->n_topic, fp);
    //fread(ctx->nz, sizeof(double), ctx->n_topic, fp);
    fclose(fp);
    return ctx;
}

Scvb0* scvb0Init(int n_topic, int n_iter, int batch_size, double alpha, double beta){
    Scvb0 *res = d_malloc(Scvb0, 1);
    
    res->n_topic = n_topic;
    res->n_iter = n_iter;
    res->batch_size = batch_size;
    res->alpha = alpha;
    res->beta = beta;
    
    return res;
}

void scvb0Free(Scvb0* ctx){
    /*free(ctx->gamma);
    free(ctx->nzHat);
    free(ctx->nPhiHat);
    free(ctx->nz);
    free(ctx->nPhi);*/
    free(ctx->Phi);
    free(ctx);
}

void scvb0EstPhi(Scvb0 *ctx, double *Phi){
    int k, v;
    for (k = 0; k < ctx->n_topic; k++) {
        double normSum = 0;
        for (v = 0; v < ctx->n_word_type; v++) {
            normSum += ctx->nPhi[v * ctx->n_topic + k] + ctx->beta;
        }
        for (v = 0; v < ctx->n_word_type; v++) {
            Phi[v * ctx->n_topic + k] = (ctx->nPhi[v * ctx->n_topic + k] + ctx->beta) / normSum;
        }
    }
}

void scvb0EstTheta(Scvb0 *ctx, double *Theta){
    int d;
    int k;
    double k_sum;
    
    for(d=0; d < ctx->n_document; d++){
        k_sum = 0.;
        for(k=0;k<ctx->n_topic;k++){
            k_sum += ctx->nTheta[d * ctx->n_topic + k] + ctx->alpha;
        }
        k_sum = 1. / k_sum ;
    
        for(k=0;k<ctx->n_topic;k++){
            Theta[d * ctx->n_topic + k] = (ctx->alpha + ctx->nTheta[d * ctx->n_topic + k] ) * k_sum;
        }
    }   
}

double perplexity(Scvb0 *ctx, int **word_indexes_ptr, unsigned short** word_counts_ptr, int *n_word_type_each_doc){
    scvb0EstPhi(ctx, ctx->nPhiHat);
    scvb0EstTheta(ctx, ctx->Theta);
    
    double log_per = 0.0;
    int N = 0;
    int d, v, k;
    
    for(d=0; d < ctx->n_document; d++){
        for(v=0; v < n_word_type_each_doc[d]; v++){
            int term = word_indexes_ptr[d][v];
            unsigned short freq = word_counts_ptr[d][v];
            double k_sum = 0.;
            for(k=0;k<ctx->n_topic;k++){
                k_sum += ctx->nPhiHat[term * ctx->n_topic + k] * ctx->Theta[d * ctx->n_topic + k];
            }
            
            log_per -= log(k_sum) * freq;
            N += freq;
        }
    }
    return exp(log_per / N);
}

void scvb0InferThread(){
}

double *scvb0TransformSingle(Scvb0 *ctx, int *doc_word, int n_word, int max_iter){
    int i, j, k;
        
    double *pzs = d_calloc(double, n_word * ctx->n_topic);
    double *pzs_new = d_malloc(double, n_word * ctx->n_topic);
    for (i=0; i < max_iter; i++) {
        for (j=0; j < n_word; j++) {
            double pzs_sum = 0;
            for (k=0; k < ctx->n_topic; k++) {
                pzs_sum += pzs[j * ctx->n_topic + k];
            }
            for (k=0; k < ctx->n_topic; k++) {
                pzs_new[j * ctx->n_topic + k] = ctx->Phi[doc_word[j] * ctx->n_topic + k] * (pzs_sum - pzs[j * ctx->n_topic + k] + ctx->alpha);
            }
        }
        for (j=0; j < n_word; j++) {
            double pzs_new_sum = 0;
            for (k=0; k < ctx->n_topic; k++) {
                pzs_new_sum += pzs_new[j * ctx->n_topic + k];
            }
            for (k=0; k < ctx->n_topic; k++) {
                pzs_new[j * ctx->n_topic + k] /= pzs_new_sum;
            }
        }
        double delta_naive = 0;
        for (j=0; j < n_word; j++) {
            for (k=0; k < ctx->n_topic; k++) {
                delta_naive += pzs_new[j * ctx->n_topic + k] - pzs[j * ctx->n_topic + k];
            }
        }
        
        memcpy(pzs, pzs_new, sizeof(double) * n_word * ctx->n_topic);
    }
    double *result = d_calloc(double, ctx->n_topic);
    for (k=0; k < ctx->n_topic; k++) {
        for (j=0; j < n_word; j++) {
            result[k] += pzs_new[j * ctx->n_topic + k];
        }
    }
    free(pzs);
    free(pzs_new);
    return result;
}

static void scvb0Infer(Scvb0 *ctx, int** word_indexes_ptr, unsigned short** word_counts_ptr, int* n_word_each_doc, int* n_word_type_each_doc,int doc_id_offset, int n_document, int *index_offset){
    int i, j, d, v, k;
    double sum_gamma;
    double batch_size_coef = 1. / n_document;
    double *temp_iter = d_malloc(double, ctx->n_topic);
    double *temp_iterA = d_malloc(double, ctx->n_topic);
    memset(ctx->nPhiHat, 0, sizeof(double) * ctx->n_word_type * ctx->n_topic);
    memset(ctx->nzHat, 0, sizeof(double) * ctx->n_topic);
    
    for(k=0; k < ctx->n_topic; k++){
        temp_iterA[k] = 1. / (ctx->nz[k] + ctx->beta * ctx->n_word_type);
    }
    for(d=0; d < n_document; d++){
        int doc_id = index_offset[d];
        double update_theta_coef = ctx->rhoTheta * n_word_each_doc[doc_id];
        
        for(i=0; i < 1; i++){
            for(v=0; v < n_word_type_each_doc[doc_id]; v++){
                int term = word_indexes_ptr[doc_id][v];
                for(k=0; k < ctx->n_topic; k++){
                    temp_iter[k] = (ctx->nPhi[term * ctx->n_topic + k] + ctx->beta) * temp_iterA[k];
                }
                
                for(j=0; j < word_counts_ptr[doc_id][v]; j++){
                    sum_gamma = 0.;
                    for(k=0; k < ctx->n_topic; k++){
                        ctx->gamma[k] = temp_iter[k] * (ctx->nTheta[doc_id * ctx->n_topic + k] + ctx->alpha);
                        sum_gamma += ctx->gamma[k];
                    }

                    for(k=0; k < ctx->n_topic; k++){
                        ctx->gamma[k] /= sum_gamma;
                        ctx->nTheta[doc_id * ctx->n_topic + k] = (1. - ctx->rhoTheta) * ctx->nTheta[doc_id * ctx->n_topic + k] + update_theta_coef * ctx->gamma[k];
                    }
                }
            }
        }
        for(v=0; v < n_word_type_each_doc[doc_id]; v++){
            int term = word_indexes_ptr[doc_id][v];
            for(k=0; k < ctx->n_topic; k++){
                temp_iter[k] = (ctx->nPhi[term * ctx->n_topic + k] + ctx->beta) * temp_iterA[k];
            }
            for(i=0; i < word_counts_ptr[doc_id][v]; i++){
                sum_gamma = 0.;
                for(k=0; k < ctx->n_topic; k++){
                    ctx->gamma[k] = temp_iter[k] *  (ctx->nTheta[doc_id * ctx->n_topic + k] + ctx->alpha);
                    sum_gamma += ctx->gamma[k];
                }
                for(k=0; k < ctx->n_topic; k++){
                    ctx->gamma[k] /= sum_gamma;
                    ctx->nTheta[doc_id * ctx->n_topic + k] = (1. - ctx->rhoTheta) * ctx->nTheta[doc_id * ctx->n_topic + k] + update_theta_coef * ctx->gamma[k];
                }
            
                for(k=0; k < ctx->n_topic; k++){
                    ctx->nPhiHat[term * ctx->n_topic + k] += ctx->n_all_word * ctx->gamma[k] * batch_size_coef;
                    ctx->nzHat[k] += ctx->n_all_word * ctx->gamma[k] * batch_size_coef;
                }
            }
        }
    }

    for(v=0; v < ctx->n_word_type; v++){
        for(k=0; k < ctx->n_topic; k++){
                ctx->nPhi[v * ctx->n_topic + k] = (1. - ctx->rhoPhi) * ctx->nPhi[v * ctx->n_topic + k] + ctx->rhoPhi * ctx->nPhiHat[v * ctx->n_topic + k];
        }
    }
    for(k=0; k < ctx->n_topic; k++){
        ctx->nz[k] = (1. - ctx->rhoPhi) * ctx->nz[k] + ctx->rhoPhi * ctx->nzHat[k];
    }
    
    free(temp_iter);
    free(temp_iterA);
}

void scvb0Fit(Scvb0 *ctx, int** word_indexes_ptr, unsigned short** word_counts_ptr, int* n_word_each_doc, int* n_word_type_each_doc,unsigned long long n_all_word, int n_document, int n_word_type){
    int i, j, k, v, d;
    ctx->n_all_word = n_all_word;
    ctx->n_word_type = n_word_type;
    ctx->n_document = n_document;
    /*
    ctx->n_word_type = -1;
    for(d=0; d < n_document; d++){
        for (v = 0; v < n_word_type_each_doc[d]; v++) {
            if (ctx->n_word_type < word_indexes_ptr[d][v]){
                ctx->n_word_type = word_indexes_ptr[d][v];
            }
        }
    }
    ctx->n_word_type++;*/

    ctx->gamma = d_malloc(double, ctx->n_topic);
    ctx->nzHat = d_malloc(double, ctx->n_topic);
    ctx->nPhiHat = d_malloc(double, ctx->n_word_type * ctx->n_topic);
    ctx->nz = d_malloc(double, ctx->n_topic);
    ctx->nPhi = d_malloc(double, ctx->n_word_type * ctx->n_topic);
    ctx->nTheta = d_malloc(double, n_document * ctx->n_topic);
    ctx->Theta = d_malloc(double, n_document * ctx->n_topic);
    
    

    for(k=0; k < ctx->n_topic; k++){
        double sum_nPhi = 0.;
        for(v=0; v < ctx->n_word_type; v++){
            ctx->nPhi[v * ctx->n_topic + k] = (double)rand() / RAND_MAX;
            sum_nPhi += ctx->nPhi[v * ctx->n_topic + k];
        }
        ctx->nz[k] = sum_nPhi;
    }
    
    for(d=0; d < n_document; d++){
        for(k=0; k < ctx->n_topic; k++){
            ctx->nTheta[d * ctx->n_topic + k] = (double)rand() / RAND_MAX;
        }
    }

    int *doc_indxes = d_malloc(int, ctx->batch_size);
    clock_t t1 = clock();
    for(i=0; i < ctx->n_iter; i++){
        if ((i % 500) == 0){
            printf("perplexity:%f\n", perplexity(ctx, word_indexes_ptr, word_counts_ptr,n_word_type_each_doc));
        }
        
        ctx->rhoPhi = 10. / pow(1000. + i, 0.9);
        ctx->rhoTheta = 1. / pow(10. + i, 0.9);
        /*ctx->rhoPhi = 0.1;
        ctx->rhoTheta = 0.1;*/
        
        for (j=0; j < ctx->batch_size; j++){
            doc_indxes[j] = xor128() % n_document;
        }
        scvb0Infer(ctx,
                    word_indexes_ptr,
                    word_counts_ptr,
                    n_word_each_doc,
                    n_word_type_each_doc,
                    j * ctx->batch_size,
                    ctx->batch_size,
                    doc_indxes);
        
        if (i % 500 == 0)
			printf("%d / %d (%.3lfsec)\n", i, ctx->n_iter, ((double)clock() - t1) / CLOCKS_PER_SEC);
        t1 = clock();
	}
	
    scvb0EstTheta(ctx, ctx->Theta);

    free(ctx->gamma);
    free(ctx->nzHat);
    free(ctx->nPhiHat);
    free(ctx->nz);
    free(ctx->nTheta);
    //free(ctx->Theta);
    ctx->Phi = d_malloc(double, ctx->n_word_type * ctx->n_topic);
    for (k = 0; k < ctx->n_topic; k++) {
        double normSum = 0;
        for (v = 0; v < ctx->n_word_type; v++) {
            normSum += ctx->nPhi[v * ctx->n_topic + k] + ctx->beta;
        }
        for (v = 0; v < ctx->n_word_type; v++) {
            ctx->Phi[v * ctx->n_topic + k] = (ctx->nPhi[v * ctx->n_topic + k] + ctx->beta) / normSum;
        }
    }
    free(ctx->nPhi);
    
}
double *scvb0FitTransform(Scvb0 *ctx, int** word_indexes_ptr, unsigned short** word_counts_ptr, int* n_word_each_doc, int* n_word_type_each_doc,unsigned long long n_all_word, int n_document, int n_word_type){
    scvb0Fit(ctx, word_indexes_ptr, word_counts_ptr, n_word_each_doc, n_word_type_each_doc, n_all_word, n_document, n_word_type);
    scvb0EstTheta(ctx, ctx->Theta);
    return ctx->Theta;
}


