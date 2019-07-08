#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "seriale.h"
#include "test.h"
#include "parallel.h"
#include "readData.h"
#include "readInitialWeight.h"


int main() {
    //initialize variable with default
    int numIn=11, numHid=5, numOut=1, numSample, numSampleTest, epochMax=600000;
    double learningRate= 0.1;
    double timeSerial=0.0, timeParallel=0.0;
    double **SweightIH, **SweightHO;
    double **PweightIH, **PweightHO;
    char ans[20];
    struct data *allData, *allDataTest;
    char *fileNameData="data.csv";
    char *fileNameDataTest="dataTest.csv";

    //read data for training
    allData=readData(numIn, numOut, &numSample, fileNameData);
    //read data for test
    allDataTest=readData(numIn, numOut, &numSampleTest, fileNameDataTest);

    //choose of parameters by input
    printf("Enter number of epoch :\n");
    scanf("%d", &epochMax);
    printf("serial o parallel or all? s/p/a\n");
    scanf("%s",ans);
    printf("Enter number of hidden neurons :\n");
    scanf("%d", &numHid);
    printf("Enter number of learning rate :\n");
    scanf("%lf", &learningRate);


    if(strcmp("a", ans)==0) { //do serial and parallel

        //read all weight for serial and parallel
        SweightIH = readInitialWeightIH(numIn, numHid);
        SweightHO = readInitialWeightHO(numHid, numOut);
        PweightIH = readInitialWeightIH(numIn, numHid);
        PweightHO = readInitialWeightHO(numHid, numOut);

        //serial training
        serial(allData, numIn, numHid, numOut, numSample, epochMax, learningRate, &timeSerial, SweightIH, SweightHO);
        //parallel training
        parallel(allData, numIn, numHid, numOut, numSample, epochMax, learningRate, &timeParallel, PweightIH, PweightHO);
        //serial test
        test(allDataTest, numIn, numHid, numOut, numSampleTest, SweightIH, SweightHO);
        //parallel test
        test(allDataTest, numIn, numHid, numOut, numSampleTest, PweightIH, PweightHO);

        //calculate delta time
        double time=timeSerial-timeParallel;

        printf("\n\nt serial=\t%.3lfs\nt parallelo=\t%.3lfs\ndifferenza=\t%.3lfs\n\n", timeSerial, timeParallel, time);

    }else if(strcmp("s", ans)==0) { //do only serial
        //read all weight for serial
        SweightIH = readInitialWeightIH(numIn, numHid);
        SweightHO = readInitialWeightHO(numHid, numOut);

        //serial training and test
        serial(allData, numIn, numHid, numOut, numSample, epochMax, learningRate, &timeSerial, SweightIH, SweightHO);
        test(allDataTest, numIn, numHid, numOut, numSampleTest, SweightIH, SweightHO);

        printf("\n\nt serial=\t%.3lfs\n\n", timeSerial);

    }else if(strcmp("p", ans)==0) { //do only parallel
        //read all weight for parallel
        PweightIH = readInitialWeightIH(numIn, numHid);
        PweightHO = readInitialWeightHO(numHid, numOut);

        //parallel training and test
        parallel(allData, numIn, numHid, numOut, numSample, epochMax, learningRate, &timeParallel, PweightIH, PweightHO);
        test(allDataTest, numIn, numHid, numOut, numSampleTest, PweightIH, PweightHO);

        printf("\n\nt parallelo=\t%.3lfs\n\n", timeParallel);
    }

    //free memory for training data
    for (int c = 0; c < numSample; c++){
        free(allData[c].out);
        free(allData[c].in);
    }
    free(allData);

    //free memory for test data
    for (int c = 0; c < numSampleTest; c++){
        free(allDataTest[c].out);
        free(allDataTest[c].in);
    }
    free(allDataTest);

    return 0;
}