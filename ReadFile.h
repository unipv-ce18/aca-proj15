#ifndef PROGETTO15_READFILE_H
#define PROGETTO15_READFILE_H

#endif //PROGETTO15_READFILE_H

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <ctype.h>

struct data
{
    double *in;
    double *out;
};


/**
 * Read a data.csv file with numIn data input per row and numOut data input per row
 * the outputs are the last ones
 * float must have dot ".", NOT comma ","
 * @param int umIn int numOut
 * @return array of struct data, the last one have in=NULL and out=NULL
 */
struct data * readData(int numIn, int numOut)
{
    int n=0, nb;
    struct data *allData;
    int allocati;  /* byte allocati */
    int dimbloc;   /* byte in un blocco */
    int dimstruct;    /* byte in un struct data */
    int usati;     /* byte struct data usati */
    nb = 100;      /* numero di dati per blocco */
    FILE *fd;
    char buf[200];
    char *fileName="data.csv";
    dimstruct = sizeof(struct data);
    dimbloc = nb * dimstruct;
    usati = 0;

    //inizializzo allData della grandezza per un blocco
    allData = (struct data *) malloc(dimbloc);
    if(allData == NULL)
    {
        perror("Not enough memory\n");
        exit(1);
    }
    allocati = dimbloc;

    allData[n].in=(double *) malloc(sizeof(double)*numIn+1);
    allData[n].out=(double *) malloc(sizeof(double)*numOut+1);
    for(int i=0; i<numIn+1;i++){
        allData[n].in[i]=0;
    }
    for(int i=0; i<numOut+1;i++){
        allData[n].out[i]=0;
    }
    n++;


    /* apre il file */
    fd=fopen(fileName, "r");
    if( fd==NULL ) {
        perror("Error, data file not found");
        exit(1);
    }

    /* leggo il file */
    while(fgets(buf, 200, fd)!=NULL) {
        usati += dimstruct;
        if(usati>allocati)
        {
            allocati += dimbloc;
            allData = (struct data *) realloc(allData, allocati);
            if(allData == NULL)
            {
                perror("Not enough memory\n");
                exit(1);
            }
        }

        int in=1, out=1;
        char *p=buf;

        allData[n].in=(double *) malloc(sizeof(double)*numIn+1);
        allData[n].out=(double *) malloc(sizeof(double)*numOut+1);
        allData[n].in[0]=0;
        allData[n].out[0]=0;

        while (*p) { // While there are more characters to process...
            if ( isdigit(*p) || ( (*p=='-'||*p=='+'||*p=='.') && isdigit(*(p+1)) )) {
                // Found a number
                double val = strtod(p, &p); // Read number
                if(in<numIn+1){
                    allData[n].in[in]=val;
                    in++;
                }
                else{
                    allData[n].out[out]=val;
                    out++;
                }
            } else {
                // Otherwise, move on to the next character.
                p++;
            }

        }


        n++;
    }


    /* chiude il file*/
    fclose(fd);

    allData[n].in=NULL;
    allData[n].out=NULL;

    return allData;
}