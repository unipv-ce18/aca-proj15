#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "seriale.h"
#include "test.h"
#include "parallel.h"
#include "readData.h"
#include "readInitialWeight.h"


int main() {
    int numIn=11, numHid=5, numOut=1, numSample, numSampleTest, epochMax=600000;
    double learningRate= 0.1;
    double timeSerial=0.0, timeParallel=0.0;
    double **SweightIH, **SweightHO;
    double **PweightIH, **PweightHO;
    char ans[20];

    struct data *allData, *allDataTest;
    char *fileNameData="data.csv";
    allData=readData(numIn, numOut, &numSample, fileNameData);

    char *fileNameDataTest="dataTest.csv";
    allDataTest=readData(numIn, numOut, &numSampleTest, fileNameDataTest);

    printf("Enter number of epoch :\n");
    scanf("%d", &epochMax);
    printf("serial o parallel or all? s/p/a\n");
    scanf("%s",ans);
    printf("Enter number of hidden neurons :\n");
    //scanf("%d", &numHid);
    printf("Enter number of learning rate :\n");
    scanf("%lf", &learningRate);

    if(strcmp("a", ans)==0) {

        SweightIH = readInitialWeightIH(numIn, numHid);
        SweightHO = readInitialWeightHO(numHid, numOut);
        PweightIH = readInitialWeightIH(numIn, numHid);
        PweightHO = readInitialWeightHO(numHid, numOut);

        serial(allData, numIn, numHid, numOut, numSample, epochMax, learningRate, &timeSerial, SweightIH, SweightHO);
        parallel(allData, numIn, numHid, numOut, numSample, epochMax, learningRate, &timeParallel, PweightIH, PweightHO);
        test(allDataTest, numIn, numHid, numOut, numSampleTest, SweightIH, SweightHO);
        test(allDataTest, numIn, numHid, numOut, numSampleTest, PweightIH, PweightHO);

        double time=timeSerial-timeParallel;
        printf("\n\nt serial=\t%.3lfs\nt parallelo=\t%.3lfs\ndifferenza=\t%.3lfs\n\n", timeSerial, timeParallel, time);

    }else if(strcmp("s", ans)==0) {

        SweightIH = readInitialWeightIH(numIn, numHid);
        SweightHO = readInitialWeightHO(numHid, numOut);

        serial(allData, numIn, numHid, numOut, numSample, epochMax, learningRate, &timeSerial, SweightIH, SweightHO);
        test(allDataTest, numIn, numHid, numOut, numSampleTest, SweightIH, SweightHO);

        printf("\n\nt serial=\t%.3lfs\n\n", timeSerial);

    }else if(strcmp("p", ans)==0) {
        PweightIH = readInitialWeightIH(numIn, numHid);
        PweightHO = readInitialWeightHO(numHid, numOut);

        parallel(allData, numIn, numHid, numOut, numSample, epochMax, learningRate, &timeParallel, PweightIH, PweightHO);
        test(allDataTest, numIn, numHid, numOut, numSampleTest, PweightIH, PweightHO);

        printf("\n\nt parallelo=\t%.3lfs\n\n", timeParallel);
    }

    for (int c = 0; c < numSample; c++){
        free(allData[c].out);
        free(allData[c].in);
    }
    free(allData);

    for (int c = 0; c < numSampleTest; c++){
        free(allDataTest[c].out);
        free(allDataTest[c].in);
    }
    free(allDataTest);

    return 0;
}