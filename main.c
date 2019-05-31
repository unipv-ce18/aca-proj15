#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "seriale.h"
#include "serialeTest.h"
#include "parallel.h"
#include "readData.h"


int main() {
    int numIn=11, numHid=4, numOut=9, numPat, numPatTest, epochMax=7000;
    double timeSeriale=0.0, timeParallel=0.0;
    char ans[20]="s";

    struct data *allData, *allDataTest;
    char *fileNameData="data.csv";
    allData=readData(numIn, numOut, &numPat, fileNameData);

    char *fileNameDataTest="dataTest.csv";
    allDataTest=readData(numIn, numOut, &numPatTest, fileNameDataTest);

    printf("Enter number of epoch :\n");
    //scanf("%d", &epochMax);
    printf("serial o parallel or all? s/p/a\n");
    //scanf("%s",ans);

    if(strcmp("a", ans)==0) {
        double ***bestWeightSeriale = seriale(allData, numIn, numHid, numOut, numPat, epochMax, &timeSeriale);
        double ***bestWeightParallel = parallel(allData, numIn, numHid, numOut, numPat, epochMax, &timeParallel);

        free(bestWeightSeriale);
        free(bestWeightParallel);

        double time=timeSeriale-timeParallel;
        printf("\n\nt seriale=\t%.3lfs\nt parallelo=\t%.3lfs\ndifferenza=\t%.3lfs\n\n", timeSeriale, timeParallel, time);

    }else if(strcmp("s", ans)==0) {
        double ***bestWeightSeriale = seriale(allData, numIn, numHid, numOut, numPat, epochMax, &timeSeriale);
        //serialeTest(allDataTest, numIn, numHid, numOut, numPatTest, bestWeightSeriale);
        free(bestWeightSeriale);

        printf("\n\nt seriale=\t%.3lfs\n\n", timeSeriale);
    }else if(strcmp("p", ans)==0) {
        double ***bestWeightParallel = parallel(allData, numIn, numHid, numOut, numPat, epochMax, &timeParallel);
        //serialeTest(allDataTest, numIn, numHid, numOut, numPatTest, bestWeightParallel);
        free(bestWeightParallel);


        printf("\n\nt parallelo=\t%.3lfs\n\n", timeParallel);
    }

    for (int c=0;c<numPat;c++){
        free(allData[c].out);
        free(allData[c].in);
    }
    free(allData);

    return 0;
}