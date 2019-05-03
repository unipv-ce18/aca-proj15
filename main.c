#include <stdio.h>
#include <stdlib.h>
#include "seriale.h"
#include "serialeTest.h"
#include "parallel.h"
#include "readData.h"


int main() {
    int numIn=11, numOut=9, numPat;

    struct data *allData, *allDataTest;
    char *fileNameData="data.csv";
    allData=readData(numIn, numOut, &numPat, fileNameData);

    char *fileNameDataTest="dataTest.csv";
    allDataTest=readData(numIn, numOut, &numPat, fileNameDataTest);

    printf("e' gia' qualcosa\n\n");

    double ***bestWeight=seriale(allData, numIn, numOut, numPat);

    printf("prova %f\n\n", bestWeight[0][4][10]);
    //parallel(allData, numIn, numOut, numPat);

    //Mi servono gli arrey dei pesi da passare al serialeTest;
    serialeTest(allDataTest, numIn, numOut, numPat, bestWeight);
    free(bestWeight);


    return 0;
}