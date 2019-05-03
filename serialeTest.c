#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include "readData.h"
#include "readInitialWeight.h"
#include "serialeTest.h"

#define NUMHID 15

#define rando() (((double)rand()/((double)RAND_MAX+1)))

int serialeTest(struct data * allData, int numIn, int numOut, int numPattern, double ***Weight) {
    printf("prova %f\n\n", Weight[0][0][1]);
    int    i, j, k, p;
    int    numHid = NUMHID;
    double SumH[numPattern+1][NUMHID+1], Hidden[numPattern+1][NUMHID+1];
    double SumO[numPattern+1][numOut+1], Output[numPattern+1][numOut+1];
    double accuracy=0;
    double precision=0;

    printf("e' gia' qualcosa2\n\n");
    for( p = 1 ; p <= numPattern ; p++ ) {    /* repeat for all the training patterns */
        for( j = 1 ; j <= numHid ; j++ ) {    /* compute hidden unit activations */

            SumH[p][j] = Weight[0][0][j];
            for( i = 1 ; i <= numIn ; i++ ) {
                SumH[p][j] += allData[p].in[i] * Weight[0][i][j] ;
            }
            Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j])) ;
        }
        printf("e' gia' qualcosa3\n\n");
        for( k = 1 ; k <= numOut ; k++ ) {    /* compute output unit activations and errors */
            printf("e' gia' qualcosa3 tris\n\n");
            SumO[p][k] = Weight[1][0][k] ;
            for( j = 1 ; j <= numHid ; j++ ) {
                printf("prova %f %d\n\n", Weight[1][j][k], j);
                printf("e' gia' qualcosa7\n\n");
                SumO[p][k] += Hidden[p][j] * Weight[1][j][k] ;
                printf("e' gia' qualcosa6\n\n");
            }
            printf("e' gia' qualcosa4\n\n");
            Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k])) ;   /* Sigmoidal Outputs VA BENE SOLO PER OUTPUT   1<=OUT<=0*/
        }
    }

    printf("e' gia' qualcosa5\n\n");
    accuracy=0; // di quanto è sbagliato
    precision=0;  //quante volte sbaglia

    for(int pat=1;pat<numPattern;pat++) {
        int class = 0;
        for (int z = 1; z <= numOut; z++) {
            if (allData[pat].out[z] < 1.1 && allData[pat].out[z] > 0.9){
                class=z;
            }
        }
        int wrong=0;
        double prob= Output[pat][class];
        for (int z = 1; z <= numOut; z++) {
            if (Output[pat][z] > prob && z!=class && Output[pat][z] > Output[pat][wrong] ){
                wrong=z;
            }
        }

        if(wrong>0){
            accuracy+=abs(wrong-class);
            precision++;
        }
    }

    accuracy=accuracy/precision;
    precision=(numPattern-precision)/numPattern;


    for( p = 1 ; p <= 10 ; p++ ) {
        fprintf(stdout, "\n%d\t", p) ;
        /*for( i = 1 ; i <= numIn ; i++ ) {
            fprintf(stdout, "%f\t", allData[p].in[i]) ;
        }*/
        fprintf(stdout, "\n\n\n") ;
        for( k = 1 ; k <= numOut ; k++ ) {
            fprintf(stdout, "\n%f\t%f\t", allData[p].out[k], Output[p][k]) ;
        }
    }

    for (int c=0;c<numPattern;c++){
        free(allData[c].out);
        free(allData[c].in);
    }
    free(allData);

    for (int c=0;c<numIn;c++){
        free(Weight[0][c]);
    }
    free(Weight[0]);

    for (int c=0;c<numHid;c++){
        free(Weight[1][c]);
    }
    free(Weight[1]);

    return 1 ;
}