#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h>
#include "parallel.h"
#include "readData.h"
#include "readInitialWeight.h"

int parallel(struct data * allData, int numIn, int numHid, int numOut, int numSample, int epochMax,
        double learningRate, double* time, double** WeightIH, double** WeightHO) {

    int    i, j, k, epoch;
    double SumH[numSample+1][numHid+1], Hidden[numSample+1][numHid+1];
    double SumO[numSample+1][numOut+1], Output[numSample+1][numOut+1];
    double DeltaO[numSample][numOut+1], PartialDeltaH[numHid+1], DeltaH[numSample][numHid+1];
    double DeltaWeightIH[numIn+1][numHid+1], DeltaWeightHO[numHid+1][numOut+1];
    double lossError, precision=0;
    double start_time = omp_get_wtime();

    omp_set_num_threads(64);
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
        lossError = 0.0 ;
        precision=0.0;


        #pragma omp parallel for private(j, i, k, PartialDeltaH) reduction(-: lossError) reduction(+: precision)
        for(int iteration = 1; iteration <= numSample; iteration++) {
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
                lossError -= (allData[iteration].out[k] * log(Output[iteration][k]) + (1.0 - allData[iteration].out[k]) * log(1.0 - Output[iteration][k]));    /*Cross-Entropy lossError UTILE PER PROBABILITY OUTPUT*/
                DeltaO[iteration][k] = allData[iteration].out[k] - Output[iteration][k];    /* Sigmoidal Outputs, Cross-Entropy lossError */
                if(fabs(allData[iteration].out[k] - Output[iteration][k]) < 0.5) precision++;
            }

            for( j = 1 ; j <= numHid ; j++ ) {    /* 'back-propagate' errors to hidden layer */
                PartialDeltaH[j] = 0.0 ;
                for( k = 1 ; k <= numOut ; k++ ) {
                    PartialDeltaH[j] += WeightHO[j][k] * DeltaO[iteration][k] ;
                }
                DeltaH[iteration][j] = PartialDeltaH[j] * Hidden[iteration][j] * (1.0 - Hidden[iteration][j]) ;
            }
        }


        for(int iteration=1; iteration<=numSample; iteration++) {
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

        for( j = 1 ; j <= numHid ; j++ ) {     /* update weights WeightIH */
            WeightIH[0][j] += learningRate*DeltaWeightIH[0][j]/numSample ;
            for( i = 1 ; i <= numIn ; i++ ) {
                WeightIH[i][j] += learningRate*DeltaWeightIH[i][j]/numSample;
            }
        }

        for( k = 1 ; k <= numOut ; k ++ ) {    /* update weights WeightHO */
            WeightHO[0][k] += learningRate*DeltaWeightHO[0][k]/numSample;
            for( j = 1 ; j <= numHid ; j++ ) {
                WeightHO[j][k] += learningRate*DeltaWeightHO[j][k]/numSample;
            }
        }
        lossError=lossError/numSample;
        precision=precision/numSample;

        if( epoch%1000 == 0 ) fprintf(stdout, "\nEpoch %-5d :   lossError = %f\tPrecision = %f", epoch, lossError, precision) ;
    }

    *time = omp_get_wtime() - start_time;

    return 1 ;
}

