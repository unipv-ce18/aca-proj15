#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include "readData.h"
#include "readInitialWeight.h"
#include "test.h"

int test(struct data *allData, int numIn, int numHid, int numOut, int numSample, double **weightIH, double **weightHO) {
    int    i, j, k;
    double SumH[numSample][numHid+1], Hidden[numSample][numHid+1];
    double SumO[numSample][numOut], Output[numSample][numOut];
    double finalOut[numSample];
    double precision=0;

    //Start Forward Propagation
    for( int iteration = 0 ; iteration < numSample ; iteration++ ) {
        for( j = 1 ; j <= numHid ; j++ ) {
            SumH[iteration][j] = weightIH[0][j]; //Bias Value
            for( i = 1 ; i <= numIn ; i++ ) {
                SumH[iteration][j] += allData[iteration].in[i] * weightIH[i][j] ;
            }

            Hidden[iteration][j] = 1.0/(1.0 + exp(-SumH[iteration][j])) ; //Sigmoidal Function
        }

        for( k = 0 ; k < numOut ; k++ ) {
            SumO[iteration][k] = weightHO[0][k] ; //Bias Value
            for( j = 1 ; j <= numHid ; j++ ) {
                SumO[iteration][k] += Hidden[iteration][j] * weightHO[j][k] ;
            }

            Output[iteration][k] = 1.0/(1.0 + exp(-SumO[iteration][k])) ;   // Sigmoidal Outputs

            //Set vector finalOut
            if (Output[iteration][k] >= 0.5) finalOut[iteration] = 1;
            else finalOut[iteration] = 0;
        }
    }

    //check the Error
    for (int iteration = 0; iteration < numSample; iteration++){
        for ( k = 0; k < numOut ; ++k) {

            if (allData[iteration].out[k] == finalOut[iteration]) precision++;
        }
    }

    printf("\n\nLa precisione finale e' del: %f", precision/numSample);

    return 1 ;
}