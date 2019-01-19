#include <stdio.h>
#include "seriale.h"
#include "TestOpenMP.h"
#include "ReadFile.h"


int main() {
    int numIn=2, numOut=1, patternsnumPat;

    struct data *allData=readData(numIn, numOut, &patternsnumPat);

    seriale(allData, numIn, numOut, patternsnumPat);

    return 0;
}