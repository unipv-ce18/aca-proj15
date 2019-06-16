#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include "readData.h"
#include "readInitialWeight.h"
#include "serialeTest.h"

#define rando() (((double)rand()/((double)RAND_MAX+1)))

int serialeTest(struct data * allData, int numIn, int numHid, int numOut, int numSample, double **weightIH, double **weightHO) {
    int    i, j, k, p;
    double SumH[numSample+1][numHid+1], Hidden[numSample+1][numHid+1];
    double SumO[numSample+1][numOut+1], Output[numSample+1][numOut+1];
    double precision=0;
    double finalOut[numSample+1];

    //printf("\n\nIl numero di Test sono: %d\n", numSample);

    //Start Forward Propagation
    for( p = 1 ; p <= numSample ; p++ ) {
        for( j = 1 ; j <= numHid ; j++ ) {
            SumH[p][j] = weightIH[0][j]; //Bias Value

            for( i = 1 ; i <= numIn ; i++ ) {
                SumH[p][j] += allData[p].in[i] * weightIH[i][j] ;
            }
            Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j])) ; //Sigmoidal Function
        }

        for( k = 1 ; k <= numOut ; k++ ) {
            SumO[p][k] = weightHO[0][k] ; //Bias Value

            for( j = 1 ; j <= numHid ; j++ ) {
                SumO[p][k] += Hidden[p][j] * weightHO[j][k] ;
            }
            Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k])) ;   // Sigmoidal Outputs

            //Set vector finalOut
            if (Output[p][k] >= 0.5) finalOut[p] = 1;
            else finalOut[p] = 0;
        }
    }

    //check the Error
    printf("\n");

    for ( p=1; p <= numSample; p++){
        for ( k = 1; k <= numOut ; ++k) {

            //printf("I Risultati sono: %f\t%f\t%f\t \n", allData[p].out[k], finalOut[p],Output[p][k]);

            if (allData[p].out[k] == finalOut[p]){
                precision++;
            }
        }
    }

    precision = precision / numSample;

    printf("\nLa precisione finale e' del: %f", precision);

    return 1 ;
}