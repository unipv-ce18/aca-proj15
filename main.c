#include <stdio.h>
#include "seriale.h"
#include "parallel.h"
#include "readData.h"


int main() {
    int numIn=11, numOut=9, numPat;

    struct data *allData;
    allData=readData(numIn, numOut, &numPat);

    printf("e' gia' qualcosa\n\n");

    double ***bestWeight=seriale(allData, numIn, numOut, numPat);
    printf("prova %f\n\n", bestWeight[0][4][10]);
    //parallel(allData, numIn, numOut, numPat);

    return 0;
}