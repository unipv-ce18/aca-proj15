#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "seriale.h"
#include "serialeTest.h"
#include "parallel.h"
#include "readData.h"
#include "readInitialWeight.h"


int main() {
    int numIn=11, numHid=5, numOut=1, numPat, numPatTest, epochMax=600000;
    double timeSeriale=0.0, timeParallel=0.0;
    double **weightIH, **weightHO;
    char ans[20]="s";

    struct data *allData, *allDataTest;
    char *fileNameData="data.csv";
    allData=readData(numIn, numOut, &numPat, fileNameData);

    char *fileNameDataTest="dataTest.csv";
    allDataTest=readData(numIn, numOut, &numPatTest, fileNameDataTest);


    weightIH=readInitialWeightIH(numIn, numHid);
    weightHO=readInitialWeightHO(numHid, numOut);

    printf("Enter number of epoch :\n");
    scanf("%d", &epochMax);
    printf("serial o parallel or all? s/p/a\n");
    //scanf("%s",ans);

    if(strcmp("a", ans)==0) {
        seriale(allData, numIn, numHid, numOut, numPat, epochMax, &timeSeriale, weightIH, weightHO);
        double ***bestWeightParallel = parallel(allData, numIn, numHid, numOut, numPat, epochMax, &timeParallel);

        free(bestWeightParallel);

        double time=timeSeriale-timeParallel;
        printf("\n\nt seriale=\t%.3lfs\nt parallelo=\t%.3lfs\ndifferenza=\t%.3lfs\n\n", timeSeriale, timeParallel, time);

    }else if(strcmp("s", ans)==0) {
        seriale(allData, numIn, numHid, numOut, numPat, epochMax, &timeSeriale, weightIH, weightHO);
        serialeTest(allDataTest, numIn, numHid, numOut, numPatTest, weightIH, weightHO);

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