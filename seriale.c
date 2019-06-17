#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h>
#include "readData.h"
#include "seriale.h"

#define rando() (((double)rand()/((double)RAND_MAX+1)))

int seriale(struct data * allData, int numIn, int numHid, int numOut, int numPattern, int epochMax, double eta, double* time,
                  double **weightIH, double **weightHO) {
    int batch=numPattern;
    int    i, j, k, p, epoch;
    double SumH[numPattern+1][numHid+1], Hidden[numPattern+1][numHid+1];
    double SumO[numPattern+1][numOut+1], Output[numPattern+1][numOut+1];
    double DeltaO[batch][numOut+1], SumDOW[numHid+1], DeltaH[batch][numHid+1];
    double DeltaWeightIH[numIn+1][numHid+1], DeltaWeightHO[numHid+1][numOut+1];
    double Error;
    double precision=0;
    double smallwt=0.5;

    /*for( j = 1 ; j <= numHid ; j++ ) {     //initialize weightIH and DeltaWeightIH
        printf("\nHidden:\t%d\n", j);
        for( i = 0 ; i <= numIn ; i++ ) {
            printf("%f\t", weightIH[i][j]);
        }
    }
    printf("\n\nDopo:\n\n");
    for( j = 1 ; j <= numHid ; j++ ) {     //initialize weightIH and DeltaWeightIH
        printf("\nHidden:\t%d\n", j);
        for( i = 0 ; i <= numIn ; i++ ) {
            DeltaWeightIH[i][j] = 0.0 ;
            weightIH[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
            printf("%f\t", weightIH[i][j]);
        }
    }
    for( k = 1 ; k <= numOut ; k ++ ) {     //initialize weightHO and DeltaWeightHO
        for( j = 0 ; j <= numHid ; j++ ) {
            DeltaWeightHO[j][k] = 0.0 ;
            weightHO[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }*/

    double start_time = omp_get_wtime();

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
        Error = 0.0 ;
        precision=0.0;
         for(int iteration=1; iteration<=batch; iteration++) {
            p=iteration;//ranpat[np+iteration];
            for (j = 1; j <= numHid; j++) {    /* compute hidden unit activations */
                SumH[p][j] = weightIH[0][j];
                for (i = 1; i <= numIn; i++) {
                    SumH[p][j] += allData[p].in[i] * weightIH[i][j];
                }
                Hidden[p][j] = 1.0 / (1.0 + exp(-SumH[p][j]));
            }
            for (k = 1; k <= numOut; k++) {    /* compute output unit activations and errors */
                SumO[p][k] = weightHO[0][k];
                for (j = 1; j <= numHid; j++) {
                    SumO[p][k] += Hidden[p][j] * weightHO[j][k];
                }

                Output[p][k] = 1.0 / (1.0 + exp(-SumO[p][k]));   /* Sigmoidal Outputs VA BENE SOLO PER OUTPUT   1<=OUT<=0*/
                Error -= (allData[p].out[k] * log(Output[p][k]) + (1.0 - allData[p].out[k]) * log(1.0 - Output[p][k]));    /*Cross-Entropy Error UTILE PER PROBABILITY OUTPUT*/
                DeltaO[iteration][k] = allData[p].out[k] - Output[p][k];    /* Sigmoidal Outputs, Cross-Entropy Error */

                if(fabs(allData[p].out[k] - Output[p][k]) < 0.49)  precision++;
            }

            for( j = 1 ; j <= numHid ; j++ ) {    /* 'back-propagate' errors to hidden layer */
                SumDOW[j] = 0.0 ;
                for( k = 1 ; k <= numOut ; k++ ) {
                    SumDOW[j] += weightHO[j][k] * DeltaO[iteration][k] ;
                }
                DeltaH[iteration][j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
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

        for( j = 1 ; j <= numHid ; j++ ) {     /* update weights weightIH */
            weightIH[0][j] += eta*DeltaWeightIH[0][j]/batch ;
            for( i = 1 ; i <= numIn ; i++ ) {
               weightIH[i][j] += eta*DeltaWeightIH[i][j]/batch;
            }
        }

        for( k = 1 ; k <= numOut ; k ++ ) {    /* update weights weightHO */
            weightHO[0][k] += eta*DeltaWeightHO[0][k]/batch;
            for( j = 1 ; j <= numHid ; j++ ) {
                weightHO[j][k] += eta*DeltaWeightHO[j][k]/batch;
            }
        }

        precision = precision/batch;

        if( epoch%1000 == 0 )
            fprintf(stdout, "\nEpoch %-5d :   Error = %f\tPrecision = %f", epoch, Error, precision) ;
    }

    *time = omp_get_wtime() - start_time;

    return 1;

}

