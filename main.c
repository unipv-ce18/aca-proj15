#include <stdio.h>
#include <stdlib.h>
#include "seriale.h"
#include "serialeTest.h"
#include "parallel.h"
#include "readData.h"


int main() {
    int numIn=11, numHid=15, numOut=9, numPat, numPatTest, epochMax=1000;
    double timeSeriale=0.0, timeParallel=0.0;

    struct data *allData, *allDataTest;
    char *fileNameData="data.csv";
    allData=readData(numIn, numOut, &numPat, fileNameData);

    char *fileNameDataTest="dataTest.csv";
    allDataTest=readData(numIn, numOut, &numPatTest, fileNameDataTest);

    double ***bestWeightSeriale=seriale(allData, numIn, numHid, numOut, numPat, epochMax, &timeSeriale);
    double ***bestWeightParallel=parallel(allData, numIn, numHid, numOut, numPat, epochMax, &timeParallel);

    for (int c=0;c<numPat;c++){
        free(allData[c].out);
        free(allData[c].in);
    }
    free(allData);


    double time=timeSeriale-timeParallel;

    printf("\n\nt seriale=\t%.3lfs\nt parallelo=\t%.3lfs\ndifferenza=\t%.3lfs\n\n", timeSeriale, timeParallel, time);

    //serialeTest(allDataTest, numIn, numHid, numOut, numPatTest, bestWeight);
    free(bestWeightSeriale);
    free(bestWeightParallel);


    return 0;
}