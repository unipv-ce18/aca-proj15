#include "readInitialWeight.h"
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <ctype.h>

double** initialWeight(int row, int colomn,  char *fileName);

double** readInitialWeightIH(int numIn, int numHid) {
    if(numIn>11 || numHid>29){
        perror("Error, input is 11 and hidden is 17");
        exit(1);
    }

    char *fileName = "initialWeightIH.csv";

    return initialWeight(numIn, numHid, fileName);
}

double** readInitialWeightHO(int numHid, int numOut) {
    if(numHid>29 || numOut>30){
        perror("Error, hidden max is 17 and out is 30");
        exit(1);
    }

    char *fileName = "initialWeightHO.csv";

    return initialWeight(numHid, numOut, fileName);
}

double** initialWeight(int row, int colomn,  char *fileName) {

    colomn++;
    row++;

    double  **Weight= (double **)malloc(row * sizeof(double*));
    for(int i = 0; i <=row; i++) Weight[i] = (double *)malloc(colomn * sizeof(double));


    FILE *fd;
    char buf[200];


    fd=fopen(fileName, "r");
    if( fd==NULL ) {
        perror("Error, data file not found");
        exit(1);
    }

    for( int j = 0 ; j < row ; j++ ) {
        if(fgets(buf, 200, fd)==NULL) {
            perror("Error, something got wrong");
            exit(1);
        }

        char *p=buf;

        for(int co=1; co<colomn; ){
            if (isdigit(*p) || ((*p == '-' || *p == '+' || *p == '.') && isdigit(*(p + 1)))) {
                Weight[j][co]= strtod(p, &p); // Read number
                co++;
            }else {
                p++;
            }
        }

    }
    fclose(fd);

    return Weight;

}

