#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h>
#include "parallel.h"
#include "readData.h"
#include "readInitialWeight.h"

#define rando() (((double)rand()/((double)RAND_MAX+1)))

int parallel(struct data * allData, int numIn, int numHid, int numOut, int numPattern, int epochMax, double* time, double** WeightIH, double** WeightHO) {
    int batch=numPattern;
    int    i, j, k, epoch;
    double SumH[numPattern+1][numHid+1], Hidden[numPattern+1][numHid+1];
    double SumO[numPattern+1][numOut+1], Output[numPattern+1][numOut+1];
    double DeltaO[batch][numOut+1], SumDOW[numHid+1], DeltaH[batch][numHid+1];
    double DeltaWeightIH[numIn+1][numHid+1], DeltaWeightHO[numHid+1][numOut+1];
    double Error, eta = 0.003;
    double precision=0;
    double smallwt=0.5;

    for( j = 1 ; j <= numHid ; j++ ) {     //initialize WeightIH and DeltaWeightIH
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
    }

    double start_time = omp_get_wtime();


    for( epoch = 0 ; epoch < epochMax ; epoch++) {    /* iterate weight updates */
        #pragma omp parallel
        {
            #pragma omp for collapse(2) nowait
            for (j = 0; j <= numHid; j++) {     /* update weights WeightIH */
                for (i = 0; i <= numIn; i++) {
                    DeltaWeightIH[i][j] = 0.0;
                }
            }
            #pragma omp for collapse(2)
            for (k = 0; k <= numOut; k++) {    /* update weights WeightHO */
                for (j = 0; j <= numHid; j++) {
                    DeltaWeightHO[j][k] = 0.0;
                }
            }
        }
        Error = 0.0 ;
        precision=0.0;


        #pragma omp parallel for private(j, i, k, SumDOW) reduction(-: Error)  //reduction(+:DeltaWeightIH)
        for(int iteration=1; iteration<=batch; iteration++) {
            for (j = 1; j <= numHid; j++) {    /* compute hidden unit activations */
                SumH[iteration][j] = WeightIH[0][j];
                for (i = 1; i <= numIn; i++) {
                    SumH[iteration][j] += allData[iteration].in[i] * WeightIH[i][j];
                }
                Hidden[iteration][j] = 1.0 / (1.0 + exp(-SumH[iteration][j]));
            }
            for (k = 1; k <= numOut; k++) {    /* compute output unit activations and errors */
                SumO[iteration][k] = WeightHO[0][k];
                for (j = 1; j <= numHid; j++) {
                    SumO[iteration][k] += Hidden[iteration][j] * WeightHO[j][k];
                }

                Output[iteration][k] = 1.0 / (1.0 + exp(-SumO[iteration][k]));   /* Sigmoidal Outputs VA BENE SOLO PER OUTPUT   1<=OUT<=0*/
                Error -= (allData[iteration].out[k] * log(Output[iteration][k]) + (1.0 - allData[iteration].out[k]) * log(1.0 - Output[iteration][k]));    /*Cross-Entropy Error UTILE PER PROBABILITY OUTPUT*/
                DeltaO[iteration][k] = allData[iteration].out[k] - Output[iteration][k];    /* Sigmoidal Outputs, Cross-Entropy Error */
                if(fabs(allData[iteration].out[k] - Output[iteration][k])<0.49)
                    precision++;
            }

            for( j = 1 ; j <= numHid ; j++ ) {    /* 'back-propagate' errors to hidden layer */
                SumDOW[j] = 0.0 ;
                for( k = 1 ; k <= numOut ; k++ ) {
                    SumDOW[j] += WeightHO[j][k] * DeltaO[iteration][k] ;
                }
                DeltaH[iteration][j] = SumDOW[j] * Hidden[iteration][j] * (1.0 - Hidden[iteration][j]) ;
            }

        }


        for(int iteration=1; iteration<=batch; iteration++) {
            for (j = 1; j <= numHid; j++) {     /* update weights WeightIH */
                DeltaWeightIH[0][j] += DeltaH[iteration][j];
                for (i = 1; i <= numIn; i++) {
                    DeltaWeightIH[i][j] += allData[iteration].in[i] * DeltaH[iteration][j];
                }
            }
            for (k = 1; k <= numOut; k++) {    /* update weights WeightHO */
                DeltaWeightHO[0][k] += DeltaO[iteration][k];
                for (j = 1; j <= numHid; j++) {
                    DeltaWeightHO[j][k] += Hidden[iteration][j] * DeltaO[iteration][k];
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


    return 1 ;
}

