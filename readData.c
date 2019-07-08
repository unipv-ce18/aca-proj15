#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <ctype.h>
#include "readData.h"

struct data * allocateFirstBlock(struct data *allData, int dimBlock);
FILE* openFile(FILE* fd, char* fileName);
struct data * allocateNewBlock(struct data *allData,int allocated);
struct data allocateAndBias(struct data data, int numIn, int numOut);
struct data readAllDigit(char* line, struct data data, int numIn);

/**
 * Read a data.csv file with numIn data input per row and numOut data input per row
 * the outputs are the last ones
 * float must have dot ".", NOT comma ","
 * @param int umIn int numOut int* patternsnumPat IT RETURN THE NUMBER OF PATTERN
 * @return array of struct data
 */
struct data * readData(int numIn, int numOut, int* numPat, char* fileName)
{
    //initialize variable
    int n=0;
    struct data *allData = NULL;
    int allocated=0;  
    int dimBlock;   
    int dimStruct;    
    int used=0;
    int nb = 10;      // # of data per block
    FILE *fd;
    char buf[200];

    dimStruct = sizeof(struct data);
    dimBlock = nb * dimStruct;

    //allocated allData for one block
    allData=allocateFirstBlock(allData,dimBlock);
    allocated+=dimBlock;

    //open of the file
    fd=openFile(fd, fileName);

    // read file
    while(fgets(buf, 200, fd)!=NULL) {
        used += dimStruct;
        // if needed allocate new block
        if(used>=allocated)
        {
            allData=allocateNewBlock(allData, allocated += dimBlock);
        }

        allData[n]=allocateAndBias(allData[n], numIn, numOut);

        allData[n]=readAllDigit(buf, allData[n], numIn);
        n++;
    }

    //close the file
    fclose(fd);

    *numPat=n;

    //sometime he fail to read the exact value of output so we standardize
    for(int p=0; p<n;p++) {
        for(int k=0; k<numOut;k++) {
            if (allData[p].out[k] < 1.1 && allData[p].out[k] > 0.9) {
                allData[p].out[k] = 1.0;
            } else {
                allData[p].out[k] = 0.0;
            }
        }
    }

    return allData;
}

struct data* allocateFirstBlock(struct data *allData, int dimBlock){
    allData = (struct data *) malloc((size_t) dimBlock);
    if(allData == NULL)
    {
        perror("Not enough memory\n");
        exit(1);
    }
    return allData;
}

FILE* openFile(FILE* fd, char* fileName){
    fd=fopen(fileName, "r");
    if( fd==NULL ) {
        char error[20]="Error, file \"";
        strcat(error, fileName);
        strcat(error, "\" not found :(");
        perror(error);
        exit(1);
    }
    return fd;
}

struct data * allocateNewBlock(struct data *allData, int allocated){
    allData = (struct data *) realloc(allData, (size_t) allocated);
    if(allData == NULL)
    {
        perror("Not enough memory\n");
        exit(1);
    }
    return allData;
}

struct data allocateAndBias(struct data data, int numIn, int numOut) {
    data.in=(double *) malloc(sizeof(double)*(numIn+1));
    data.out=(double *) malloc(sizeof(double)*(numOut));
    if(data.out == NULL || data.in==NULL)
    {
        perror("Not enough memory\n");
        exit(1);
    }
    data.in[0]=1;
    return data;
}


struct data readAllDigit(char* line, struct data data, int numIn){
    int in=1, out=0;
    char *p=line;

    while (*p) { // While there are more characters to process...
        if ( isdigit(*p) || ( (*p=='-'||*p=='+'||*p=='.') && isdigit(*(p+1)) )) {
            // Found a number
            double val = strtod(p, &p); // Read number
            if(in<numIn+1){
                data.in[in]=val;
                in++;
            }
            else{
                data.out[out]=val;
                out++;
            }
        } else {
            // Otherwise, move on to the next character.
            p++;
        }
    }
    return data;
}