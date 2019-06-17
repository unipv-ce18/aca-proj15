#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h>
#include "readData.h"
#include "seriale.h"

int serial(struct data *allData, int numIn, int numHid, int numOut, int numSample, int epochMax, double learningRate,
           double *time,
           double **weightIH, double **weightHO) {
    
    int    i, j, k, epoch;
    double SumH[numSample+1][numHid+1], Hidden[numSample+1][numHid+1];
    double SumO[numSample+1][numOut+1], Output[numSample+1][numOut+1];
    double DeltaO[numSample+1][numOut+1], PartialDeltaH[numHid+1], DeltaH[numSample+1][numHid+1];
    double DeltaWeightIH[numIn+1][numHid+1], DeltaWeightHO[numHid+1][numOut+1];
    double lossError, precision=0;
    double start_time = omp_get_wtime();
    double serial_time;
    double serial_t = 0.0;


    for( epoch = 0 ; epoch < epochMax ; epoch++) {    /* iterate weight updates */

        for( j = 0 ; j <= numHid ; j++ ) {     /* update weights weightIH */
            for( i = 0 ; i <= numIn ; i++ ) {
                DeltaWeightIH[i][j] =0.0;
            }
        }
        for( k = 0 ; k <= numOut ; k ++ ) {    /* update weights weightHO */
            for( j = 0 ; j <= numHid ; j++ ) {
                DeltaWeightHO[j][k] =0.0;
            }
        }

        lossError = 0.0 ;
        precision=0.0;

        for(int iteration = 1; iteration <= numSample; iteration++) {

            for (j = 1; j <= numHid; j++) {    /* compute hidden unit activations */
                SumH[iteration][j] = weightIH[0][j];
                for (i = 1; i <= numIn; i++) {
                    SumH[iteration][j] += allData[iteration].in[i] * weightIH[i][j];
                }
                Hidden[iteration][j] = 1.0 / (1.0 + exp(-SumH[iteration][j]));
            }

            for (k = 1; k <= numOut; k++) {    /* compute output unit activations and errors */
                SumO[iteration][k] = weightHO[0][k];
                for (j = 1; j <= numHid; j++) {
                    SumO[iteration][k] += Hidden[iteration][j] * weightHO[j][k];
                }

                Output[iteration][k] = 1.0 / (1.0 + exp(-SumO[iteration][k]));   /* Sigmoidal Outputs VA BENE SOLO PER OUTPUT   1<=OUT<=0*/
                lossError -= (allData[iteration].out[k] * log(Output[iteration][k]) + (1.0 - allData[iteration].out[k]) * log(1.0 - Output[iteration][k]));    /*Cross-Entropy lossError UTILE PER PROBABILITY OUTPUT*/
                DeltaO[iteration][k] = allData[iteration].out[k] - Output[iteration][k];    /* Sigmoidal Outputs, Cross-Entropy lossError */

                if(fabs(allData[iteration].out[k] - Output[iteration][k]) < 0.49)  precision++;
            }

            for( j = 1 ; j <= numHid ; j++ ) {    /* 'back-propagate' errors to hidden layer */
                PartialDeltaH[j] = 0.0 ;
                for( k = 1 ; k <= numOut ; k++ ) {
                    PartialDeltaH[j] += weightHO[j][k] * DeltaO[iteration][k] ;
                }
                DeltaH[iteration][j] = PartialDeltaH[j] * Hidden[iteration][j] * (1.0 - Hidden[iteration][j]) ;
            }
        }

        serial_time = omp_get_wtime();

        for(int iteration = 1; iteration <= numSample; iteration++) {
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

        for( j = 1 ; j <= numHid ; j++ ) {     /* update weights weightIH */
            weightIH[0][j] += learningRate*DeltaWeightIH[0][j]/numSample ;
            for( i = 1 ; i <= numIn ; i++ ) {
               weightIH[i][j] += learningRate*DeltaWeightIH[i][j]/numSample;
            }
        }

        for( k = 1 ; k <= numOut ; k ++ ) {    /* update weights weightHO */
            weightHO[0][k] += learningRate*DeltaWeightHO[0][k]/numSample;
            for( j = 1 ; j <= numHid ; j++ ) {
                weightHO[j][k] += learningRate*DeltaWeightHO[j][k]/numSample;
            }
        }

        lossError = lossError/numSample;
        precision = precision/numSample;

        if( epoch%1000 == 0 ) fprintf(stdout, "\nEpoch %-5d :   lossError = %f\tPrecision = %f", epoch, lossError, precision) ;
        serial_t+=omp_get_wtime()-serial_time;
    }

    printf("\nTempo non parallelizzabile=%lf\n", serial_t);
    *time = omp_get_wtime() - start_time;

    return 1;
}

