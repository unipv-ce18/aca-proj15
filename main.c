#include <stdio.h>
#include "seriale.h"
#include "serialeTest.h"
#include "parallel.h"
#include "readData.h"


int main() {
    int numIn=11, numOut=9, numPat;

    struct data *allData, *allDataTest;
    char *fileNameData="data.csv";
    allData=readData(numIn, numOut, &numPat, fileNameData);

    printf("e' gia' qualcosa\n\n");

    double ***bestWeight=seriale(allData, numIn, numOut, numPat);

    printf("prova %f\n\n", bestWeight[0][4][10]);
    //parallel(allData, numIn, numOut, numPat);

    char *fileNameDataTest="dataTest.csv";
    allDataTest=readData(numIn, numOut, &numPat, fileNameDataTest);

    //Mi servono gli arrey dei pesi da passare al serialeTest;
    serialeTest(allDataTest, numIn, numOut, numPat, bestWeight[0], bestWeight[1]);


    return 0;
}