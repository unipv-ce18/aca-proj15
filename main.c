#include <stdio.h>
#include <stdlib.h>
#include "seriale.h"
#include "serialeTest.h"
#include "parallel.h"
#include "readData.h"


int main() {
    int numIn=11, numOut=9, numPat, numPatTest;

    struct data *allData, *allDataTest;
    char *fileNameData="data.csv";
    allData=readData(numIn, numOut, &numPat, fileNameData);

    char *fileNameDataTest="dataTest.csv";
    allDataTest=readData(numIn, numOut, &numPatTest, fileNameDataTest);

    printf("e' gia' qualcosa\n\n");

    //double ***bestWeight=seriale(allData, numIn, numOut, numPat);
    double ***bestWeight=parallel(allData, numIn, numOut, numPat);

    serialeTest(allDataTest, numIn, numOut, numPatTest, bestWeight);
    free(bestWeight);


    return 0;
}