#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include "readFile.h"
#include "seriale.h"


#define NUMHID 7

#define rando() (((double)rand()/((double)RAND_MAX+1)))

#define max(a,b)  (((a) > (b)) ? (a) : (b))

int seriale(struct data * allData, int numIn, int numOut, int numPattern) {
    int    i, j, k, p, np, op, ranpat[numPattern+1], epoch;
    int    NumHidden = NUMHID;
    double SumH[numPattern+1][NUMHID+1], WeightIH[numIn+1][NUMHID+1], Hidden[numPattern+1][NUMHID+1];
    double SumO[numPattern+1][numOut+1], WeightHO[NUMHID+1][numOut+1], Output[numPattern+1][numOut+1];
    double DeltaO[numOut+1], SumDOW[NUMHID+1], DeltaH[NUMHID+1];
    double DeltaWeightIH[numIn+1][NUMHID+1], DeltaWeightHO[NUMHID+1][numOut+1];
    double Error, eta = 0.5, alpha = 0.1, smallwt = 0.5;
    double accuracy, minAccuracy=1.0;
    double sensitivity, maxSensitivity=0;

    for( j = 1 ; j <= NumHidden ; j++ ) {    /* initialize WeightIH and DeltaWeightIH */
        for( i = 0 ; i <= numIn ; i++ ) {
            DeltaWeightIH[i][j] = 0.0 ;
            WeightIH[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }
    for( k = 1 ; k <= numOut ; k ++ ) {    /* initialize WeightHO and DeltaWeightHO */
        for( j = 0 ; j <= NumHidden ; j++ ) {
            DeltaWeightHO[j][k] = 0.0 ;
            WeightHO[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }

    for( epoch = 0 ; epoch < 20000 ; epoch++) {    /* iterate weight updates */
        for( p = 1 ; p <= numPattern ; p++ ) {    /* randomize order of training patterns */
            ranpat[p] = p ;
        }
        for( p = 1 ; p <= numPattern ; p++) {
            np = (int) (p + rando() * (numPattern + 1 - p ));
            op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
        }
        Error = 0.0 ;
        for( np = 1 ; np <= numPattern ; np++ ) {    /* repeat for all the training patterns */
            p = ranpat[np];
            for( j = 1 ; j <= NumHidden ; j++ ) {    /* compute hidden unit activations */
                SumH[p][j] = WeightIH[0][j] ;
                for( i = 1 ; i <= numIn ; i++ ) {
                    SumH[p][j] += allData[p].in[i] * WeightIH[i][j] ;
                }
                Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j])) ;
            }
            for( k = 1 ; k <= numOut ; k++ ) {    /* compute output unit activations and errors */
                SumO[p][k] = WeightHO[0][k] ;
                for( j = 1 ; j <= NumHidden ; j++ ) {
                    SumO[p][k] += Hidden[p][j] * WeightHO[j][k] ;
                }

//                Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k])) ;   /* Sigmoidal Outputs VA BENE SOLO PER OUTPUT   1<=OUT<=0*/
/*                double expSum=0;
                for( int h = 1 ; h <= numOut ; h++ ) {expSum+=SumO[p][h];}
                Output[p][k] = exp(-SumO[p][k])/expSum;   /* Soft max  utile se si hanno più output*/
                Output[p][k] = SumO[p][k];    /*  Linear Outputs */
//                Output[p][k] = log(1+exp(SumO[p][k]));
//                Output[p][k] = max(0, SumO[p][k]);

                Error += 0.5 * (allData[p].out[k] - Output[p][k]) * (allData[p].out[k] - Output[p][k]) ;   /* SSE */
//                Error -= ( allData[p].out[k] * log( Output[p][k] ) + ( 1.0 - allData[p].out[k] ) * log( 1.0 - Output[p][k] ) ) ;    /*Cross-Entropy Error UTILE PER PROBABILITY OUTPUT*/

//                DeltaO[k] = (allData[p].out[k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;   /* Sigmoidal Outputs, SSE */
//                DeltaO[k] = allData[p].out[k] - Output[p][k];    /* Sigmoidal Outputs, Cross-Entropy Error */
                DeltaO[k] = allData[p].out[k] - Output[p][k];    /* Linear Outputs, SSE */
            }
            for( j = 1 ; j <= NumHidden ; j++ ) {    /* 'back-propagate' errors to hidden layer */
                SumDOW[j] = 0.0 ;
                for( k = 1 ; k <= numOut ; k++ ) {
                    SumDOW[j] += WeightHO[j][k] * DeltaO[k] ;
                }
                DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
            }
            for( j = 1 ; j <= NumHidden ; j++ ) {     /* update weights WeightIH */
                DeltaWeightIH[0][j] = eta * DeltaH[j] + alpha * DeltaWeightIH[0][j] ;
                WeightIH[0][j] += DeltaWeightIH[0][j] ;
                for( i = 1 ; i <= numIn ; i++ ) {
                    DeltaWeightIH[i][j] = eta * allData[p].in[i] * DeltaH[j] + alpha * DeltaWeightIH[i][j];
                    WeightIH[i][j] += DeltaWeightIH[i][j] ;
                }
            }
            for( k = 1 ; k <= numOut ; k ++ ) {    /* update weights WeightHO */
                DeltaWeightHO[0][k] = eta * DeltaO[k] + alpha * DeltaWeightHO[0][k] ;
                WeightHO[0][k] += DeltaWeightHO[0][k] ;
                for( j = 1 ; j <= NumHidden ; j++ ) {
                    DeltaWeightHO[j][k] = eta * Hidden[p][j] * DeltaO[k] + alpha * DeltaWeightHO[j][k] ;
                    WeightHO[j][k] += DeltaWeightHO[j][k] ;
                }
            }
        }

        accuracy=0;
        sensitivity=0;
        for (int z=1;z<numPattern;z++){
            double deltaAccuracy=fabs(allData[z].out[1]- (roundf((float) (Output[z][1] * 10)) / 10));

            if(deltaAccuracy<0.1) {
                sensitivity++;
            } else{
                accuracy+=deltaAccuracy;
            }
        }

        accuracy=accuracy/(numPattern-1-sensitivity);
        sensitivity=sensitivity/(numPattern-1);

        if(minAccuracy>accuracy)
            minAccuracy=accuracy;

        if(maxSensitivity<sensitivity)
            maxSensitivity=sensitivity;

        if( epoch%100 == 0 )
            fprintf(stdout, "\nEpoch %-5d :   Error = %f, accuracy = %f, sensitivity = %f", epoch, Error/(numPattern-1), accuracy, sensitivity) ;
        if( accuracy < (0.01) )
            break ;  /* stop learning when 'near enough' */
    }

    fprintf(stdout, "\n\nMinima accurateza  %-4f\t", minAccuracy) ;
    fprintf(stdout, "\n\nMassima sensitivita' %-4f\t", maxSensitivity) ;
    fprintf(stdout, "\n\nNETWORK DATA - EPOCH %d\n\nPat\t", epoch) ;   /* print network outputs */
    /*for( i = 1 ; i <= numIn ; i++ ) {
        fprintf(stdout, "Input%-4d\t", i) ;
    }
    for( k = 1 ; k <= numOut ; k++ ) {
        fprintf(stdout, "Target%-4d\tOutput%-4d\t", k, k) ;
    }
    for( p = 1 ; p <= 100 ; p++ ) {
        fprintf(stdout, "\n%d\t", p) ;
        for( i = 1 ; i <= numIn ; i++ ) {
            fprintf(stdout, "%f\t", allData[p].in[i]) ;
        }
        for( k = 1 ; k <= numOut ; k++ ) {
            fprintf(stdout, "%f\t%f\t", allData[p].out[k], Output[p][k]) ;
        }
    }*/

    for (int c=0;c<numPattern;c++){
        free(allData[c].out);
        free(allData[c].in);
    }
    free(allData);

    return 1 ;
}

