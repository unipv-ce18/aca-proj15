#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h>
#include "readData.h"
#include "seriale.h"
#include "readInitialWeight.h"

#define rando() (((double)rand()/((double)RAND_MAX+1)))

double*** seriale(struct data * allData, int numIn, int numHid, int numOut, int numPattern, int epochMax, double* time) {
    int batch=numPattern;
    int    i, j, k, p, epoch;
    double SumH[numPattern+1][numHid+1], **WeightIH, Hidden[numPattern+1][numHid+1];
    double SumO[numPattern+1][numOut+1], **WeightHO, Output[numPattern+1][numOut+1];
    double DeltaO[numOut+1], SumDOW[numHid+1], DeltaH[numHid+1];
    double DeltaWeightIH[numIn+1][numHid+1], DeltaWeightHO[numHid+1][numOut+1];
    double Error, eta = 0.00003;
    double precision=0;
    double smallwt=0.5;

    WeightIH=readInitialWeightIH(numIn, numHid);
    WeightHO=readInitialWeightHO(numHid, numOut);

    /*for( j = 1 ; j <= numHid ; j++ ) {     //initialize WeightIH and DeltaWeightIH
        for( i = 0 ; i <= numIn ; i++ ) {
            DeltaWeightIH[i][j] = 0.0 ;
            WeightIH[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }
    for( k = 1 ; k <= numOut ; k ++ ) {     //initialize WeightHO and DeltaWeightHO
        for( j = 0 ; j <= numHid ; j++ ) {
            DeltaWeightHO[j][k] = 0.0 ;
            WeightHO[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }*/

    double start_time = omp_get_wtime();

    for( i = 0 ; i <= numIn ; i++ ) {
        for( j = 1 ; j <= numHid ; j++ ) {
            DeltaWeightIH[i][j] = 0.0 ;
        }
    }


    for( j = 0 ; j <= numHid ; j++ ) {
        for( k = 1 ; k <= numOut ; k ++ ) {
            DeltaWeightHO[j][k] = 0.0 ;
        }
    }

    for( epoch = 0 ; epoch < epochMax ; epoch++) {    /* iterate weight updates */
        for (k = 1; k <= numOut; k++) {
            DeltaO[k] = 0.0;
        }
        for (j = 1; j <= numHid; j++) {
            DeltaH[j] = 0.0;
        }

        for( j = 0 ; j <= numHid ; j++ ) {     /* update weights WeightIH */
            for( i = 0 ; i <= numIn ; i++ ) {
                DeltaWeightIH[i][j] =0.0;
            }
        }
        for( k = 0 ; k <= numOut ; k ++ ) {    /* update weights WeightHO */
            for( j = 0 ; j <= numHid ; j++ ) {
                DeltaWeightHO[j][k] =0.0;
            }
        }
        Error = 0.0 ;
        precision=0.0;
         for(int iteration=1; iteration<batch; iteration++) {
            p=iteration;//ranpat[np+iteration];
            for (j = 1; j <= numHid; j++) {    /* compute hidden unit activations */
                SumH[p][j] = WeightIH[0][j];
                for (i = 1; i <= numIn; i++) {
                    SumH[p][j] += allData[p].in[i] * WeightIH[i][j];
                }
                Hidden[p][j] = 1.0 / (1.0 + exp(-SumH[p][j]));
            }
            for (k = 1; k <= numOut; k++) {    /* compute output unit activations and errors */
                SumO[p][k] = WeightHO[0][k];
                for (j = 1; j <= numHid; j++) {
                    SumO[p][k] += Hidden[p][j] * WeightHO[j][k];
                }

                Output[p][k] = 1.0 / (1.0 + exp(-SumO[p][k]));   /* Sigmoidal Outputs VA BENE SOLO PER OUTPUT   1<=OUT<=0*/
                Error -= (allData[p].out[k] * log(Output[p][k]) + (1.0 - allData[p].out[k]) * log(1.0 - Output[p][k]));    /*Cross-Entropy Error UTILE PER PROBABILITY OUTPUT*/
                DeltaO[k] += allData[p].out[k] - Output[p][k];    /* Sigmoidal Outputs, Cross-Entropy Error */
                //fprintf(stdout, "\nVero%f :   predizione = %f", allData[p].out[k], Output[p][k]) ;
                if(allData[p].out[k] - Output[p][k]<0.4)
                    precision++;
            }

            for( j = 1 ; j <= numHid ; j++ ) {    /* 'back-propagate' errors to hidden layer */
                SumDOW[j] = 0.0 ;
                for( k = 1 ; k <= numOut ; k++ ) {
                    SumDOW[j] += WeightHO[j][k] * DeltaO[k] ;
                }
                DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
            }

             for( j = 1 ; j <= numHid ; j++ ) {     /* update weights WeightIH */
                 DeltaWeightIH[0][j] += DeltaH[j];
                 for( i = 1 ; i <= numIn ; i++ ) {
                     DeltaWeightIH[i][j] += allData[p].in[i] * DeltaH[j];
                 }
             }
             for( k = 1 ; k <= numOut ; k ++ ) {    /* update weights WeightHO */
                 DeltaWeightHO[0][k] += DeltaO[k];
                 for( j = 1 ; j <= numHid ; j++ ) {
                     DeltaWeightHO[j][k] +=Hidden[p][j] * DeltaO[k];
                 }
             }

        }
         Error=Error/batch;

        for( j = 1 ; j <= numHid ; j++ ) {     /* update weights WeightIH */
            WeightIH[0][j] += eta*DeltaWeightIH[0][j]/batch ;
            for( i = 1 ; i <= numIn ; i++ ) {
               WeightIH[i][j] += eta*DeltaWeightIH[i][j]/batch;
            }
        }

        for( k = 1 ; k <= numOut ; k ++ ) {    /* update weights WeightHO */
            WeightHO[0][k] += eta*DeltaWeightHO[0][k]/batch;
            for( j = 1 ; j <= numHid ; j++ ) {
                WeightHO[j][k] += eta*DeltaWeightHO[j][k]/batch;
            }
        }

        precision=precision/batch;

        if( epoch%100 == 0 )
            fprintf(stdout, "\nEpoch %-5d :   Error = %f\tPrecision = %f", epoch, Error, precision) ;
    }

    *time = omp_get_wtime() - start_time;



    double  ***bestWeight= (double ***)malloc(2 * sizeof(double**));
    bestWeight[0]= (double **)malloc(numIn * sizeof(double*));
    bestWeight[1]= (double **)malloc(numHid * sizeof(double*));
    for(int i = 0; i <=numIn+1; i++) bestWeight[0][i] = (double *)malloc(numHid * sizeof(double));
    for(int i = 0; i <=numHid+1; i++) bestWeight[1][i] = (double *)malloc(numOut * sizeof(double));
    bestWeight[0]=WeightIH;
    bestWeight[1]=WeightHO;


    return bestWeight ;

}

