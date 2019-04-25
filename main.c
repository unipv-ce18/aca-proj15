#include <stdio.h>
#include "seriale.h"
#include "parallel.h"
#include "readData.h"


int main() {
    int numIn=11, numOut=9, numPat;

    struct data *allData;
    allData=readData(numIn, numOut, &numPat);

    printf("e' gia' qualcosa\n\n");

    seriale(allData, numIn, numOut, numPat);
    //parallel(allData, numIn, numOut, numPat);

    return 0;
}