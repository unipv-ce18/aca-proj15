#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <ctype.h>
#include "readData.h"

struct data * allocateFirstBlock(struct data *allData, int dimBlock);
struct data firstDataAsBias(struct data allData, int numIn, int numOut);
FILE* openFile(FILE* fd, char* fileName);
struct data * allocateNewBlock(struct data *allData,int allocati);
struct data firstDataInAndOutAsBias(struct data data, int numIn, int numOut);
struct data readAllDigit(char* line, struct data data, int numIn);

/**
 * Read a data.csv file with numIn data input per row and numOut data input per row
 * the outputs are the last ones
 * float must have dot ".", NOT comma ","
 * @param int umIn int numOut int* patternsnumPat IT RETURN THE NUMBER OF PATTERN
 * @return array of struct data
 */
struct data * readData(int numIn, int numOut, int* numPat)
{
    int n=0;
    struct data *allData = NULL;
    int allocati=0;  /* byte allocati */
    int dimBlock;   /* byte in un blocco */
    int dimstruct;    /* byte in un struct data */
    int usati=0;     /* byte struct data usati */
    int nb = 10;      /* numero di dati per blocco */

    FILE *fd;
    char buf[200];
    char *fileName="data.csv";

    dimstruct = sizeof(struct data);
    dimBlock = nb * dimstruct;


    allData=allocateFirstBlock(allData,dimBlock); //alloco allData della grandezza per un blocco
    allocati+=dimBlock;

    allData[n]=firstDataAsBias(allData[n], numIn, numOut);
    n++;

    fd=openFile(fd, fileName);

    /* leggo il file */
    while(fgets(buf, 200, fd)!=NULL) {
        usati += dimstruct;
        if(usati>=allocati)
        {
            allData=allocateNewBlock(allData, allocati += dimBlock);
        }

        allData[n]=firstDataInAndOutAsBias(allData[n], numIn, numOut);

        allData[n]=readAllDigit(buf, allData[n], numIn);
        n++;
    }

    /* chiude il file*/
    fclose(fd);

    *numPat=n-1;

    for(int p=0; p<n;p++) {
        for(int k=0; k<=numOut;k++) {
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

struct data firstDataAsBias(struct data allData, int numIn, int numOut){
    allData.in=(double *) malloc(sizeof(double)*(numIn+1));
    allData.out=(double *) malloc(sizeof(double)*(numOut+1));
    if(allData.out == NULL || allData.in==NULL)
    {
        perror("Not enough memory\n");
        exit(1);
    }

    for(int i=0; i<numIn+1;i++){
        allData.in[i]=0;
    }
    for(int i=0; i<numOut+1;i++){
        allData.out[i]=0;
    }
    return allData;
}

FILE* openFile(FILE* fd, char* fileName){
    fd=fopen(fileName, "r");
    if( fd==NULL ) {
        perror("Error, data file not found");
        exit(1);
    }
    return fd;
}

struct data * allocateNewBlock(struct data *allData, int allocati){
    allData = (struct data *) realloc(allData, (size_t) allocati);
    if(allData == NULL)
    {

        perror("Not enough memory\n");
        exit(1);
    }
    return allData;
}

struct data firstDataInAndOutAsBias(struct data data, int numIn, int numOut) {
    data.in=(double *) malloc(sizeof(double)*(numIn+1));
    data.out=(double *) malloc(sizeof(double)*(numOut+1));
    if(data.out == NULL || data.in==NULL)
    {
        perror("Not enough memory\n");
        exit(1);
    }
    data.in[0]=0;
    data.out[0]=0;
    return data;
}


struct data readAllDigit(char* line, struct data data, int numIn){
    int in=1, out=1;
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