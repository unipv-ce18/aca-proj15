#include <stdio.h>
#include "seriale.h"
#include "ReadFile.h"


int main() {
    int numIn=2, numOut=1, numPat;

    struct data *allData;
    allData=readData(numIn, numOut, &numPat);

    printf("e' già qualcosa\n\n");

    seriale(allData, numIn, numOut, numPat);

    return 0;
}