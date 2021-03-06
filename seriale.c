#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h>
#include "readData.h"
#include "seriale.h"

int serial(struct data *allData, int numIn, int numHid, int numOut, int numSample, int epochMax, double learningRate, double *time, double **WeightIH, double **WeightHO) {
    //initialize variable
    int    i, j, k, epoch;
    double SumH[numSample][numHid+1], Hidden[numSample][numHid+1];
    double SumO[numSample][numOut], Output[numSample][numOut];
    double DeltaO[numSample][numOut], PartialDeltaH[numHid+1], DeltaH[numSample][numHid+1];
    double DeltaWeightIH[numIn+1][numHid+1], DeltaWeightHO[numHid+1][numOut];
    double lossError, precision=0;
    double start_time = omp_get_wtime();
    //double serial_time;
    //double serial_t = 0.0;

    for(int sample = 0; sample < numSample; sample++) {
        Hidden[sample][0]=1.0;
    }

    for( epoch = 0 ; epoch < epochMax ; epoch++) {

        for (i = 0; i <= numIn; i++) { // initialize  DeltaWeightIH to 0
            for (j = 0; j <= numHid; j++) {
                DeltaWeightIH[i][j] = 0.0;
            }
        }

        for (j = 0; j <= numHid; j++) { // initialize  DeltaWeightHO to 0
            for (k = 0; k < numOut; k++) {
                DeltaWeightHO[j][k] = 0.0;
            }
        }

        lossError = 0.0 ;
        precision=0.0;

        for(int iteration = 0; iteration < numSample; iteration++) {

            for (j = 1; j <= numHid; j++) {    // compute hidden unit activations
                SumH[iteration][j] = WeightIH[0][j];
                for (i = 1; i <= numIn; i++) {
                    SumH[iteration][j] += allData[iteration].in[i] * WeightIH[i][j];
                }
                Hidden[iteration][j] = 1.0 / (1.0 + exp(-SumH[iteration][j]));  // Sigmoidal Hidden
            }

            for (k = 0; k < numOut; k++) {    // compute output unit activations and errors
                SumO[iteration][k] = WeightHO[0][k];
                for (j = 1; j <= numHid; j++) {
                    SumO[iteration][k] += Hidden[iteration][j] * WeightHO[j][k];
                }

                Output[iteration][k] = 1.0 / (1.0 + exp(-SumO[iteration][k]));   // Sigmoidal Outputs
                lossError -= (allData[iteration].out[k] * log(Output[iteration][k]) + (1.0 - allData[iteration].out[k]) * log(1.0 - Output[iteration][k]));    // Cross-Entropy lossError
                DeltaO[iteration][k] = allData[iteration].out[k] - Output[iteration][k];    // Delta for Sigmoidal Outputs, Cross-Entropy lossError

                if(fabs(allData[iteration].out[k] - Output[iteration][k]) < 0.5)  precision++;
            }

            for( j = 1 ; j <= numHid ; j++ ) {    // 'back-propagate' errors to hidden layer
                PartialDeltaH[j] = 0.0 ;
                for( k = 0 ; k < numOut ; k++ ) {
                    PartialDeltaH[j] += WeightHO[j][k] * DeltaO[iteration][k] ;
                }
                DeltaH[iteration][j] = PartialDeltaH[j] * Hidden[iteration][j] * (1.0 - Hidden[iteration][j]) ;
            }
        }

        //serial_time = omp_get_wtime();

        for (int iteration = 0; iteration < numSample; iteration++) {

            for (i = 0; i <= numIn; i++) { // compute deltaWeightIH
                for (j = 1; j <= numHid; j++) {
                    DeltaWeightIH[i][j] += allData[iteration].in[i] * DeltaH[iteration][j];
                }
            }

            for (j = 0; j <= numHid; j++) { // compute deltaWeightHO
                for (k = 0; k < numOut; k++) {
                    DeltaWeightHO[j][k] += Hidden[iteration][j] * DeltaO[iteration][k];
                }
            }
        }

        for (i = 0; i <= numIn; i++) { // update weights WeightIH
            for (j = 1; j <= numHid; j++) {
                WeightIH[i][j] += learningRate * DeltaWeightIH[i][j] / numSample;
            }
        }

        for (j = 0; j <= numHid; j++) { // update weights WeightHO
            for (k = 0; k < numOut; k++) {
                WeightHO[j][k] += learningRate * DeltaWeightHO[j][k] / numSample;
            }
        }

        lossError = lossError/numSample;
        precision = precision/numSample;

        if( epoch%500 == 0 ) {
            fprintf(stdout, "\nEpoch %-5d :\tLossError = %f\tPrecision of training data = %f", epoch, lossError, precision);
        }
        //serial_t+=omp_get_wtime()-serial_time;
    }

    //printf("\nTime code not parallelizable=%lf\n", serial_t);
    *time = omp_get_wtime() - start_time;

    return 1;
}

