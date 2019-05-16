#include <stdio.h>
#include <stdlib.h>
#include "seriale.h"
#include "serialeTest.h"
#include "parallel.h"
#include "readData.h"


int main() {
    int numIn=11, numHid=15, numOut=9, numPat, numPatTest;

    struct data *allData, *allDataTest;
    char *fileNameData="data.csv";
    allData=readData(numIn, numOut, &numPat, fileNameData);

    char *fileNameDataTest="dataTest.csv";
    allDataTest=readData(numIn, numOut, &numPatTest, fileNameDataTest);

    printf("e' gia' qualcosa\n\n");

    //double ***bestWeight=seriale(allData, numIn, numHid, numOut, numPat);
    double ***bestWeight=parallel(allData, numIn, numHid, numOut, numPat);

    serialeTest(allDataTest, numIn, numHid, numOut, numPatTest, bestWeight);
    free(bestWeight);


    return 0;
}