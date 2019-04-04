#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h>
#include "parallel.h"
#include "readData.h"
#include "readInitialWeight.h"


#define NUMHID 7

#define rando() (((double)rand()/((double)RAND_MAX+1)))

int parallel(struct data * allData, int numIn, int numOut, int numPattern) {

    int    i, j, k, p, np, op, ranpat[numPattern+1], epoch;
    int    NumHidden = NUMHID;
    double SumH[numPattern+1][NUMHID+1], **WeightIH, Hidden[numPattern+1][NUMHID+1];
    double SumO[numPattern+1][numOut+1], **WeightHO, Output[numPattern+1][numOut+1];
    double DeltaO[numOut+1], SumDOW[NUMHID+1], DeltaH[NUMHID+1];
    double DeltaWeightIH[numIn+1][NUMHID+1], DeltaWeightHO[NUMHID+1][numOut+1];
    double Error, eta = 0.02, alpha = 0.1;
    double accuracy=0, minAccuracy=10.0;
    double sensitivity=0, maxSensitivity=0;


    WeightIH=readInitialWeightIH(numIn, NumHidden);
    WeightHO= readInitialWeightHO(NumHidden, numOut);

    double start = omp_get_wtime();

    #pragma omp parallel
    {

        #pragma omp for collapse(2) nowait
        for (i = 0; i <= numIn; i++) {
            for (j = 1; j <= NumHidden; j++) {
                DeltaWeightIH[i][j] = 0.0;
            }
        }

        #pragma omp for collapse(2)
        for (j = 0; j <= NumHidden; j++) {
            for (k = 1; k <= numOut; k++) {
                DeltaWeightHO[j][k] = 0.0;
            }
        }
    }

    for( epoch = 0 ; epoch < 5000 ; epoch++) {

            /* iterate weight updates */
        //#pragma omp parallel
        {
            //#pragma omp for
            for (p = 1; p <= numPattern; p++) {    /* randomize order of training patterns */
                ranpat[p] = p;
            }
        }
            //#pragma omp for reduction(+: np)  private(op)
            for (p = 1; p <= numPattern; p++) {
                np = (int) (p + rando() * (numPattern + 1 - p));
                op = ranpat[p];
                ranpat[p] = ranpat[np];
                ranpat[np] = op;
            }


        Error = 0.0 ;
        #pragma omp parallel for private(p, j, i, k, SumDOW, DeltaO, DeltaH) firstprivate( DeltaWeightIH, DeltaWeightHO) lastprivate(DeltaWeightIH, DeltaWeightHO)
        for( np = 1 ; np <= numPattern ; np++ ) {    /* repeat for all the training patterns */
            p = ranpat[np];
            double **WeightIH2=WeightIH;
            double **WeightHO2=WeightHO;



//#pragma omp parallel for private (i, j)
            for (j = 1; j <= NumHidden; j++) {    /* compute hidden unit activations */
                SumH[p][j] = WeightIH2[0][j];
                double sum = 0;
                for (i = 1; i <= numIn; i++) {
                    sum += allData[p].in[i] * WeightIH2[i][j];
                }
                SumH[p][j] += sum;
                Hidden[p][j] = 1.0 / (1.0 + exp(-SumH[p][j]));
            }

            double newError = 0;
          //  #pragma omp parallel for reduction(+:newError) private(j)
            for (k = 1; k <= numOut; k++) {    /* compute output unit activations and errors */
                SumO[p][k] = WeightHO2[0][k];
                //#pragma omp parallel for
                for (j = 1; j <= NumHidden; j++) {
                    SumO[p][k] += Hidden[p][j] * WeightHO2[j][k];
                }

                Output[p][k] = 1.0 / (1.0 + exp(-SumO[p][k]));   /* Sigmoidal Outputs*/
                newError += (allData[p].out[k] * log(Output[p][k]) +
                             (1.0 - allData[p].out[k]) * log(1.0 - Output[p][k]));    /*Cross-Entropy Error*/


                DeltaO[k] = allData[p].out[k] - Output[p][k];    /* Sigmoidal Outputs, Cross-Entropy Error */



            #pragma omp atomic
            Error -= newError;
        }
            for (j = 1; j <= NumHidden; j++) {    /* 'back-propagate' errors to hidden layer */
                SumDOW[j] = 0.0;
                for (k = 1; k <= numOut; k++) {
                    SumDOW[j] += WeightHO2[j][k] * DeltaO[k];
                }
                DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]);
            }
            for (j = 1; j <= NumHidden; j++) {     /* update weights WeightIH */
                DeltaWeightIH[0][j] = eta * DeltaH[j] + alpha * DeltaWeightIH[0][j];
                WeightIH2[0][j] += DeltaWeightIH[0][j];
                for (i = 1; i <= numIn; i++) {
                    DeltaWeightIH[i][j] = eta * allData[p].in[i] * DeltaH[j] + alpha * DeltaWeightIH[i][j];
                    WeightIH2[i][j] += DeltaWeightIH[i][j];
                }
            }
            for (k = 1; k <= numOut; k++) {    /* update weights WeightHO */
                DeltaWeightHO[0][k] = eta * DeltaO[k] + alpha * DeltaWeightHO[0][k];
                WeightHO2[0][k] += DeltaWeightHO[0][k];
                for (j = 1; j <= NumHidden; j++) {
                    DeltaWeightHO[j][k] = eta * Hidden[p][j] * DeltaO[k] + alpha * DeltaWeightHO[j][k];
                    WeightHO2[j][k] += DeltaWeightHO[j][k];
                }
            }

//#pragma omp critical
            {
                WeightIH = WeightIH2;
                WeightHO = WeightHO2;
            }


        }

        accuracy=0;
        sensitivity=0;

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
                if (Output[pat][z] > prob && z!=class){
                    wrong++;
                }
            }

            if(wrong>0){
                accuracy+=wrong;
                sensitivity++;
            }
        }

        accuracy=accuracy/(sensitivity);
        sensitivity=(numPattern-sensitivity)/numPattern;


        if(minAccuracy>accuracy)
            minAccuracy=accuracy;

        if(maxSensitivity<sensitivity)
            maxSensitivity=sensitivity;

        if( epoch%100 == 0 )
            fprintf(stdout, "\nEpoch %-5d :   Error = %f\tminAcc=%-4f,\tmaxSens=%-4f,\t", epoch, Error/numPattern, accuracy, sensitivity) ;
        //if( accuracy < (0.01) )
        //break ;  /* stop learning when 'near enough' */
    }


    //printf(stdout, "\n\nNETWORK DATA - EPOCH %d\n\nPat\t", epoch) ;   /* print network outputs */
    /*for( i = 1 ; i <= numIn ; i++ ) {
        fprintf(stdout, "Input%-4d\t", i) ;
    }
    for( k = 1 ; k <= numOut ; k++ ) {
        fprintf(stdout, "Target%-4d\tOutput%-4d\t", k, k) ;
    }*/
    for( p = 1 ; p <= 10 ; p++ ) {
        fprintf(stdout, "\n\n%d\t", p) ;
        /*for( i = 1 ; i <= numIn ; i++ ) {
            fprintf(stdout, "%f\t", allData[p].in[i]) ;
        }*/
        for( k = 1 ; k <= numOut ; k++ ) {
            fprintf(stdout, "\n%f\t%f\t", allData[p].out[k], Output[p][k]) ;
        }
    }
    double time = omp_get_wtime()-start;

    //if((minAccuracy-1.3081969)<0.000001 && (maxSensitivity-0.6460289)<0.000001) {
        fprintf(stdout, "\ntempo=%-4f", time);
        double oldTime = 30.396000;
        double deltaT = oldTime - time;
        if (deltaT > 0) {
            fprintf(stdout, "\nMIGLIORATO di %-4f", deltaT);
        } else {
            fprintf(stdout, "\nPEGGIORATO di %-4f", deltaT);
        }

    //}else {

        fprintf(stdout, "\nERRORE\nminAcc=%-4f,\t", minAccuracy);
        fprintf(stdout, "maxSens=%-4f", maxSensitivity);
    //}

    for (int c=0;c<numPattern;c++){
        free(allData[c].out);
        free(allData[c].in);
    }
    free(allData);

    return 1 ;
}

