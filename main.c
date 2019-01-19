#include <stdio.h>
#include "seriale.h"
#include "TestOpenMP.h"
#include "ReadFile.h"


int main() {
    int numIn=11, numOut=1, numPat;

    struct data *allData=readData(numIn, numOut, &numPat);

    seriale(allData, numIn, numOut, numPat);

    return 0;
}