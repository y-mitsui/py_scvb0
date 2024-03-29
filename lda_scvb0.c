#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "lda_scvb0.h"

#define d_malloc(type, numb) (type *)debug_malloc(sizeof(type) * numb)
#define d_calloc(type, numb) (type *)debug_calloc(sizeof(type), numb)

void *debug_malloc(size_t size)
{
    void *p = malloc(size);
    if (p == NULL)
    {
        fprintf(stderr, "out of memory %luB", size);
        exit(0);
    }
    return p;
}

void *debug_calloc(size_t size, size_t numb)
{
    void *p = calloc(size, numb);
    if (p == NULL)
    {
        fprintf(stderr, "out of memory %luB", size);
        exit(0);
    }
    return p;
}

unsigned long xor128()
{
    static unsigned long x = 123456789, y = 362436069, z = 521288629, w = 88675123;
    unsigned long t;
    t = (x ^ (x << 11));
    x = y;
    y = z;
    z = w;
    return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
}

void scvb0Save(Scvb0 *ctx, const char *path)
{
    FILE *fp;

    if (!(fp = fopen(path, "wb")))
    {
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

    fwrite(ctx->Theta, sizeof(double), ctx->n_document * ctx->n_topic, fp);
    fwrite(ctx->Phi, sizeof(double), ctx->n_word_type * ctx->n_topic, fp);
    fclose(fp);
}

Scvb0 *scvb0Load(const char *path)
{
    FILE *fp;
    Scvb0 *ctx = d_malloc(Scvb0, 1);

    if (!(fp = fopen(path, "rb")))
    {
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

    ctx->Phi = d_malloc(double, ctx->n_word_type * ctx->n_topic);
    ctx->Theta = d_malloc(double, ctx->n_document * ctx->n_topic);

    fread(ctx->Theta, sizeof(double), ctx->n_document * ctx->n_topic, fp);
    fread(ctx->Phi, sizeof(double), ctx->n_word_type * ctx->n_topic, fp);
    fclose(fp);

    return ctx;
}

int scvb0Sample(const char *path, int ***word_indexes_r, unsigned short ***word_counts_r, int **n_word_type_each_doc_r, int **n_word_each_doc_r, unsigned long long *n_all_word_r, int *n_document_r, int *n_word_type_r)
{
    int n_document;
    FILE *fp_word_indexes = fopen(path, "rb");
    if (!fp_word_indexes)
        return -1;
    fread(&n_document, sizeof(int), 1, fp_word_indexes);
    int **word_indexes = d_malloc(int *, n_document);
    unsigned short **word_counts = d_malloc(unsigned short *, n_document);
    int *n_word_type_each_doc = d_malloc(int, n_document);
    int *n_word_each_doc = d_malloc(int, n_document);
    unsigned long long n_all_word = 0;
    int n_word_type = 0;
    int i, j;
    for (i = 0; i < n_document; i++)
    {
        fread(&n_word_type_each_doc[i], sizeof(int), 1, fp_word_indexes);
        word_indexes[i] = d_malloc(int, n_word_type_each_doc[i]);
        word_counts[i] = d_malloc(unsigned short, n_word_type_each_doc[i]);
        for (j = 0; j < n_word_type_each_doc[i]; j++)
        {
            fread(&word_indexes[i][j], sizeof(int), 1, fp_word_indexes);
            fread(&word_counts[i][j], sizeof(unsigned short), 1, fp_word_indexes);
            n_word_each_doc[i] += word_counts[i][j];
            n_all_word += word_counts[i][j];
            if (n_word_type < word_indexes[i][j])
                n_word_type = word_indexes[i][j];
        }
    }
    fclose(fp_word_indexes);

    n_word_type++;
    *word_indexes_r = word_indexes;
    *word_counts_r = word_counts;
    *n_word_type_each_doc_r = n_word_type_each_doc;
    *n_word_each_doc_r = n_word_each_doc;
    *n_all_word_r = n_all_word;
    *n_document_r = n_document;
    *n_word_type_r = n_word_type;
    return 0;
}

Scvb0 *scvb0Init(int n_topic, int n_iter, int batch_size, int n_thread, double alpha, double beta)
{
    Scvb0 *res = d_calloc(Scvb0, 1);

    res->n_topic = n_topic;
    res->n_iter = n_iter;
    res->batch_size = batch_size;
    res->n_thread = n_thread;
    res->alpha = alpha;
    res->beta = beta;

    return res;
}

void scvb0Free(Scvb0 *ctx)
{
    free(ctx->Phi);
    free(ctx->Theta);
    free(ctx);
}

void scvb0EstPhi(Scvb0 *ctx, double *Phi)
{
    int k, v;
    for (k = 0; k < ctx->n_topic; k++)
    {
        double normSum = 0;
        for (v = 0; v < ctx->n_word_type; v++)
        {
            normSum += ctx->nPhi[v * ctx->n_topic + k] + ctx->beta;
        }
        for (v = 0; v < ctx->n_word_type; v++)
        {
            Phi[v * ctx->n_topic + k] = (ctx->nPhi[v * ctx->n_topic + k] + ctx->beta) / normSum;
        }
    }
}

void scvb0EstTheta(Scvb0 *ctx, double *Theta)
{
    int d;
    int k;
    double k_sum;

    for (d = 0; d < ctx->n_document; d++)
    {
        k_sum = 0.;
        for (k = 0; k < ctx->n_topic; k++)
        {
            k_sum += ctx->nTheta[d * ctx->n_topic + k] + ctx->alpha;
        }
        k_sum = 1. / k_sum;

        for (k = 0; k < ctx->n_topic; k++)
        {
            Theta[d * ctx->n_topic + k] = (ctx->alpha + ctx->nTheta[d * ctx->n_topic + k]) * k_sum;
        }
    }
}

double perplexity(Scvb0 *ctx, int **word_indexes_ptr, unsigned short **word_counts_ptr, int *n_word_type_each_doc)
{
    scvb0EstPhi(ctx, ctx->nPhiHat);
    scvb0EstTheta(ctx, ctx->Theta);

    double log_per = 0.0;
    int N = 0;
    int d, v, k;

    for (d = 0; d < ctx->n_document; d++)
    {
        for (v = 0; v < n_word_type_each_doc[d]; v++)
        {
            int term = word_indexes_ptr[d][v];
            unsigned short freq = word_counts_ptr[d][v];
            double k_sum = 0.;
            for (k = 0; k < ctx->n_topic; k++)
            {
                k_sum += ctx->nPhiHat[term * ctx->n_topic + k] * ctx->Theta[d * ctx->n_topic + k];
            }

            log_per -= log(k_sum) * freq;
            N += freq;
        }
    }
    return exp(log_per / N);
}

double *scvb0TransformSingle(Scvb0 *ctx, int *doc_word, int n_word, int max_iter)
{
    int i, j, k;

    double *pzs = d_calloc(double, n_word * ctx->n_topic);
    double *pzs_new = d_malloc(double, n_word * ctx->n_topic);
    for (i = 0; i < max_iter; i++)
    {
        for (j = 0; j < n_word; j++)
        {
            double pzs_sum = 0;
            for (k = 0; k < ctx->n_topic; k++)
            {
                pzs_sum += pzs[j * ctx->n_topic + k];
            }
            for (k = 0; k < ctx->n_topic; k++)
            {
                pzs_new[j * ctx->n_topic + k] = ctx->Phi[doc_word[j] * ctx->n_topic + k] * (pzs_sum - pzs[j * ctx->n_topic + k] + ctx->alpha);
            }
        }
        for (j = 0; j < n_word; j++)
        {
            double pzs_new_sum = 0;
            for (k = 0; k < ctx->n_topic; k++)
            {
                pzs_new_sum += pzs_new[j * ctx->n_topic + k];
            }
            for (k = 0; k < ctx->n_topic; k++)
            {
                pzs_new[j * ctx->n_topic + k] /= pzs_new_sum;
            }
        }
        double delta_naive = 0;
        for (j = 0; j < n_word; j++)
        {
            for (k = 0; k < ctx->n_topic; k++)
            {
                delta_naive += pzs_new[j * ctx->n_topic + k] - pzs[j * ctx->n_topic + k];
            }
        }

        memcpy(pzs, pzs_new, sizeof(double) * n_word * ctx->n_topic);
    }
    double *result = d_calloc(double, ctx->n_topic);
    for (k = 0; k < ctx->n_topic; k++)
    {
        for (j = 0; j < n_word; j++)
        {
            result[k] += pzs_new[j * ctx->n_topic + k];
        }
    }
    free(pzs);
    free(pzs_new);
    return result;
}

static void scvb0Infer(Scvb0 *ctx, int **word_indexes_ptr, unsigned short **word_counts_ptr, int *n_word_each_doc, int *n_word_type_each_doc, int doc_id_offset, int n_document, int *index_offset)
{
    int i, j, d, v, k;
    double sum_gamma;
    double batch_size_coef = 1. / n_document;
    double *temp_iter = d_malloc(double, ctx->n_topic);
    double *temp_iterA = d_malloc(double, ctx->n_topic);

    double *nPhiHat = d_calloc(double, ctx->n_word_type * ctx->n_topic);
    double *nzHat = d_calloc(double, ctx->n_word_type * ctx->n_topic);

    double *gamma = d_malloc(double, ctx->n_topic);

    for (k = 0; k < ctx->n_topic; k++)
    {
        temp_iterA[k] = 1. / (ctx->nz[k] + ctx->beta * ctx->n_word_type);
    }
    for (d = 0; d < n_document; d++)
    {
        int doc_id = index_offset[d];
        double update_theta_coef = ctx->rhoTheta * n_word_each_doc[doc_id];

        for (i = 0; i < 1; i++)
        {
            for (v = 0; v < n_word_type_each_doc[doc_id]; v++)
            {
                int term = word_indexes_ptr[doc_id][v];

                for (k = 0; k < ctx->n_topic; k++)
                {
                    temp_iter[k] = (ctx->nPhi[term * ctx->n_topic + k] + ctx->beta) * temp_iterA[k];
                }

                for (j = 0; j < word_counts_ptr[doc_id][v]; j++)
                {
                    sum_gamma = 0.;
                    for (k = 0; k < ctx->n_topic; k++)
                    {
                        gamma[k] = temp_iter[k] * (ctx->nTheta[doc_id * ctx->n_topic + k] + ctx->alpha);
                        sum_gamma += gamma[k];
                    }

                    for (k = 0; k < ctx->n_topic; k++)
                    {
                        gamma[k] /= sum_gamma;
                        ctx->nTheta[doc_id * ctx->n_topic + k] = (1. - ctx->rhoTheta) * ctx->nTheta[doc_id * ctx->n_topic + k] + update_theta_coef * gamma[k];
                    }
                }
            }
        }
        for (v = 0; v < n_word_type_each_doc[doc_id]; v++)
        {
            int term = word_indexes_ptr[doc_id][v];
            for (k = 0; k < ctx->n_topic; k++)
            {
                temp_iter[k] = (ctx->nPhi[term * ctx->n_topic + k] + ctx->beta) * temp_iterA[k];
            }
            for (i = 0; i < word_counts_ptr[doc_id][v]; i++)
            {
                sum_gamma = 0.;
                for (k = 0; k < ctx->n_topic; k++)
                {
                    gamma[k] = temp_iter[k] * (ctx->nTheta[doc_id * ctx->n_topic + k] + ctx->alpha);
                    sum_gamma += gamma[k];
                }
                for (k = 0; k < ctx->n_topic; k++)
                {
                    gamma[k] /= sum_gamma;
                    ctx->nTheta[doc_id * ctx->n_topic + k] = (1. - ctx->rhoTheta) * ctx->nTheta[doc_id * ctx->n_topic + k] + update_theta_coef * gamma[k];
                }

                for (k = 0; k < ctx->n_topic; k++)
                {
                    nPhiHat[term * ctx->n_topic + k] += ctx->n_all_word * gamma[k] * batch_size_coef;
                    nzHat[k] += ctx->n_all_word * gamma[k] * batch_size_coef;
                }
            }
        }
    }

    for (v = 0; v < ctx->n_word_type; v++)
    {
        for (k = 0; k < ctx->n_topic; k++)
        {
            ctx->nPhi[v * ctx->n_topic + k] = (1. - ctx->rhoPhi) * ctx->nPhi[v * ctx->n_topic + k] + ctx->rhoPhi * nPhiHat[v * ctx->n_topic + k];
        }
    }
    for (k = 0; k < ctx->n_topic; k++)
    {
        ctx->nz[k] = (1. - ctx->rhoPhi) * ctx->nz[k] + ctx->rhoPhi * nzHat[k];
    }

    free(nPhiHat);
    free(nzHat);
    free(gamma);
    free(temp_iter);
    free(temp_iterA);
}

void scvb0Fit(Scvb0 *ctx, int **word_indexes_ptr, unsigned short **word_counts_ptr, int *n_word_each_doc, int *n_word_type_each_doc, unsigned long long n_all_word, int n_document, int n_word_type)
{
    int k, v, d;

    if (ctx->Theta != NULL)
    { // 初回呼び出しでなければメモリー解放
        free(ctx->Theta);
        free(ctx->Phi);
    }

    ctx->n_all_word = n_all_word;
    ctx->n_word_type = n_word_type;
    ctx->n_document = n_document;
    ctx->gamma = d_malloc(double, ctx->n_topic);
    ctx->nzHat = d_malloc(double, ctx->n_topic);
    ctx->nPhiHat = d_malloc(double, ctx->n_word_type * ctx->n_topic);
    ctx->nz = d_malloc(double, ctx->n_topic);
    ctx->nPhi = d_malloc(double, ctx->n_word_type * ctx->n_topic);
    ctx->nTheta = d_malloc(double, n_document * ctx->n_topic);
    ctx->Theta = d_malloc(double, n_document * ctx->n_topic);

    int *doc_indxes = d_malloc(int, ctx->batch_size);
    for (k = 0; k < ctx->n_topic; k++)
    {
        double sum_nPhi = 0.;
        for (v = 0; v < ctx->n_word_type; v++)
        {
            ctx->nPhi[v * ctx->n_topic + k] = (double)rand() / RAND_MAX;
            sum_nPhi += ctx->nPhi[v * ctx->n_topic + k];
        }
        ctx->nz[k] = sum_nPhi;
    }

    for (d = 0; d < n_document; d++)
    {
        for (k = 0; k < ctx->n_topic; k++)
        {
            ctx->nTheta[d * ctx->n_topic + k] = (double)rand() / RAND_MAX;
        }
    }

    for (int i = 0; i < ctx->n_iter; i++)
    {
        ctx->rhoPhi = 10. / pow(1000. + i, 0.9);
        ctx->rhoTheta = 1. / pow(10. + i, 0.9);
        // 文書をランダムにサンプリング
        for (int j = 0; j < ctx->batch_size; j++)
        {
            doc_indxes[j] = xor128() % n_document;
        }
        scvb0Infer(ctx,
                   word_indexes_ptr,
                   word_counts_ptr,
                   n_word_each_doc,
                   n_word_type_each_doc,
                   0,
                   ctx->batch_size,
                   doc_indxes);
    }

    scvb0EstTheta(ctx, ctx->Theta);

    free(ctx->gamma);
    free(ctx->nzHat);
    free(ctx->nPhiHat);
    free(ctx->nz);
    free(ctx->nTheta);

    ctx->Phi = d_malloc(double, ctx->n_word_type * ctx->n_topic);
    for (k = 0; k < ctx->n_topic; k++)
    {
        double normSum = 0;
        for (v = 0; v < ctx->n_word_type; v++)
        {
            normSum += ctx->nPhi[v * ctx->n_topic + k] + ctx->beta;
        }
        for (v = 0; v < ctx->n_word_type; v++)
        {
            ctx->Phi[v * ctx->n_topic + k] = (ctx->nPhi[v * ctx->n_topic + k] + ctx->beta) / normSum;
        }
    }
    free(ctx->nPhi);
}

double *scvb0FitTransform(Scvb0 *ctx, int **word_indexes_ptr, unsigned short **word_counts_ptr, int *n_word_each_doc, int *n_word_type_each_doc, unsigned long long n_all_word, int n_document, int n_word_type)
{
    scvb0Fit(ctx, word_indexes_ptr, word_counts_ptr, n_word_each_doc, n_word_type_each_doc, n_all_word, n_document, n_word_type);
    scvb0EstTheta(ctx, ctx->Theta);
    return ctx->Theta;
}

void scvb0FitFile(Scvb0 *ctx, const char *path)
{
    int **word_indexes;
    unsigned short **word_counts;
    int *n_word_type_each_doc, *n_word_each_doc;
    int n_document, n_word_type;
    unsigned long long n_all_word;

    scvb0Sample(path, &word_indexes, &word_counts, &n_word_type_each_doc, &n_word_each_doc, &n_all_word, &n_document, &n_word_type);
    scvb0Fit(ctx, word_indexes, word_counts, n_word_each_doc, n_word_type_each_doc, n_all_word, n_document, n_word_type);
    free(word_indexes);
    free(word_counts);
    free(n_word_type_each_doc);
    free(n_word_each_doc);
}
