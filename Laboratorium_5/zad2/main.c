#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <unistd.h>
#include <time.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_sf_bessel.h>

void naive_multiplication(int size, double **A, double **B, double **C){
    for (int i = 0 ; i < size ; i++){
        for (int j = 0 ; j < size ; j++){
            for (int k = 0 ; k < size ; k ++){
                C[i][j] += A[i][k]*B[k][j];
            }
        }     
    }
}

void better_multiplication(int size, double **A, double **B, double **C){
    for (int i = 0; i < size; i++){
        for (int k = 0; k < size; k ++){
            for (int j = 0; j < size; j++){
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }    
}

void blas(int size, double *a, double *b, double *c){
    gsl_matrix_view A = gsl_matrix_view_array(a, size, size);
    gsl_matrix_view B = gsl_matrix_view_array(b, size, size);
    gsl_matrix_view C = gsl_matrix_view_array(c, size, size);
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, &A.matrix, &B.matrix, 0.0, &C.matrix);
}

int main(int argc, char *argv[]){

    double **A ,**B, **C , *a ,*b ,*c , naive_time , better_time ,blas_time;
    struct timespec start, end;
    clock_t res = sysconf(_SC_CLK_TCK);
    srand(time(NULL));   

    FILE *file = fopen("results.csv","a");
    fprintf(file, "size,naive,better,blas\n");

    for(int i = 2 ; i <= 102 ; i +=10){
        A =calloc(i,sizeof(double*));
        B =calloc(i,sizeof(double*));
        C =calloc(i,sizeof(double*));
        a = calloc(i*i, sizeof(double));
        b = calloc(i*i, sizeof(double));
        c = calloc(i*i, sizeof(double));
        for(int j = 0 ; j < i ; j++){
            A[j] = calloc(i,sizeof(double));
            B[j] = calloc(i,sizeof(double));
            C[j] = calloc(i,sizeof(double));
        }
        for(int j=0; j<10 ;j++){
            for (int x = 0; x < i; x ++){
            for (int y = 0; y < i; y ++){
                A[x][y] = rand()/RAND_MAX;
                B[x][y] = rand()/RAND_MAX;
                }
            }
        
 

            for(int j=0; j < i*i ; j++){
                a[j] = A[j/i][j%i];
                b[j] = B[j/i][j%i];
            }


            clock_gettime(CLOCK_MONOTONIC, &start);
            naive_multiplication(i,A,B,C); 
            clock_gettime(CLOCK_MONOTONIC, &end);
            naive_time = (double) (end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)/10e9;

            clock_gettime(CLOCK_MONOTONIC, &start);
            better_multiplication(i,A,B,C); 
            clock_gettime(CLOCK_MONOTONIC, &end);
            better_time = (double) (end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)/10e9;

            clock_gettime(CLOCK_MONOTONIC, &start);
            blas(i,a,b,c); 
            clock_gettime(CLOCK_MONOTONIC, &end);
            blas_time = (double) (end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)/10e9;

            fprintf(file, "%d,%f,%f,%f\n" , i,naive_time,better_time,blas_time);
        
        }

        for (int j = 0; j < i; j++){
            free(A[j]);
            free(B[j]);
            free(C[j]);
        }

        free(A);
        free(B);
        free(C);
        free(a);
        free(b);
        free(c);

    }

    fclose(file);

}

