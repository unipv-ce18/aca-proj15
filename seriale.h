#ifndef PROGETTO15_SERIALE_H
#define PROGETTO15_SERIALE_H

    #include "structData.h"

    int serial(struct data *allData, int numIn, int numHid, int numOut, int numSample, int epochMax,
               double learningRate, double *time, double **weightIH, double **weightHO);

#endif //PROGETTO15_SERIALE_H