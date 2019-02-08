#include <stdio.h>
#include "seriale.h"
#include "readFile.h"


int main() {
    int numIn=11, numOut=1, numPat;

    struct data *allData;
    allData=readData(numIn, numOut, &numPat);

    printf("e' gia' qualcosa\n\n");

    seriale(allData, numIn, numOut, numPat);

    return 0;
}