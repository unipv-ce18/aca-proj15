#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h>
#include "parallel.h"
#include "readData.h"
#include "readInitialWeight.h"

int parallel(struct data * allData, int numIn, int numHid, int numOut, int numSample, int epochMax, double learningRate, double* time, double** WeightIH, double** WeightHO) {

    int    i, j, k, epoch;
    double SumH[numSample][numHid+1], Hidden[numSample][numHid+1];
    double SumO[numSample][numOut], Output[numSample][numOut];
    double DeltaO[numSample][numOut], PartialDeltaH[numHid+1], DeltaH[numSample][numHid+1];
    double DeltaWeightIH[numIn+1][numHid+1], DeltaWeightHO[numHid+1][numOut];
    double lossError, precision=0;
    double start_time = omp_get_wtime();

    for(int sample = 0; sample < numSample; sample++) {
        Hidden[sample][0]=1.0;
    }

    for( epoch = 0 ; epoch < epochMax ; epoch++) {
        #pragma omp parallel
        {
            #pragma omp for collapse(2) nowait
            for (i = 0; i <= numIn; i++) { // initialize  DeltaWeightIH
                for (j = 0; j <= numHid; j++) {
                    DeltaWeightIH[i][j] = 0.0;
                }
            }
            #pragma omp for collapse(2)
            for (j = 0; j <= numHid; j++) { // initialize  DeltaWeightHO
                for (k = 0; k < numOut; k++) {
                    DeltaWeightHO[j][k] = 0.0;
                }
            }
        }

        lossError = 0.0;
        precision = 0.0;


#pragma omp parallel for private(j, i, k, PartialDeltaH) reduction(-: lossError) reduction(+: precision)
        for (int iteration = 0; iteration < numSample; iteration++) {
            for (j = 1; j <= numHid; j++) {    // compute hidden unit activations
                SumH[iteration][j] = WeightIH[0][j];
                for (i = 1; i <= numIn; i++) {
                    SumH[iteration][j] += allData[iteration].in[i] * WeightIH[i][j];
                }
                Hidden[iteration][j] = 1.0 / (1.0 + exp(-SumH[iteration][j])); // Sigmoidal Hidden
            }
            for (k = 0; k < numOut; k++) {    // compute output unit activations and errors
                SumO[iteration][k] = WeightHO[0][k];
                for (j = 1; j <= numHid; j++) {
                    SumO[iteration][k] += Hidden[iteration][j] * WeightHO[j][k];
                }

                Output[iteration][k] = 1.0 / (1.0 + exp(-SumO[iteration][k]));   /* Sigmoidal Outputs VA BENE SOLO PER OUTPUT   1<=OUT<=0*/
                lossError -= (allData[iteration].out[k] * log(Output[iteration][k]) + (1.0 - allData[iteration].out[k]) * log(1.0 - Output[iteration][k]));    /*Cross-Entropy lossError UTILE PER PROBABILITY OUTPUT*/
                DeltaO[iteration][k] = allData[iteration].out[k] - Output[iteration][k];    /* Sigmoidal Outputs, Cross-Entropy lossError */
                if (fabs(allData[iteration].out[k] - Output[iteration][k]) < 0.5) precision++;
            }

            for (j = 1; j <= numHid; j++) {    // 'back-propagate' errors to hidden layer
                PartialDeltaH[j] = 0.0;
                for (k = 0; k < numOut; k++) {
                    PartialDeltaH[j] += WeightHO[j][k] * DeltaO[iteration][k];
                }
                DeltaH[iteration][j] = PartialDeltaH[j] * Hidden[iteration][j] * (1.0 - Hidden[iteration][j]);
            }
        }

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

        lossError=lossError/numSample;
        precision=precision/numSample;


        if( epoch%500 == 0 ){
            fprintf(stdout, "\nEpoch %-5d :\tLossError = %f\tPrecision of training data = %f", epoch, lossError, precision) ;
        }
    }

    *time = omp_get_wtime() - start_time;

    return 1 ;
}

