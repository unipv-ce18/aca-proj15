#ifndef PROGETTO15_READFILE_H
#define PROGETTO15_READFILE_H

#endif //PROGETTO15_READFILE_H

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <ctype.h>

struct data
{
    float *in;
    float *out;
};


/**
 * Read a data.csv file with numIn data input per row and numOut data input per row
 * the outputs are the last ones
 * float must have dot ".", NOT comma ","
 * @param int umIn int numOut
 * @return si vedrà
 */
int readData(int numIn, int numOut)
{
    int i=0, n=0, x, nb;
    struct data *allData;
    int allocati;  /* byte allocati */
    int dimbloc;   /* byte in un blocco */
    int dimint;    /* byte in un struct data */
    int usati;     /* byte struct data usati */
    nb = 100;      /* numero di dati per blocco */
    FILE *fd;
    char buf[200];
    char *res;
    char *fileName="data.csv";
    dimint = sizeof(struct data);
    dimbloc = nb * dimint;
    usati = 0;

    //inizializzo allData della grandezza per un blocco
    allData = (struct data *) malloc(dimbloc);
    if(allData == NULL)
    {
        perror("Not enough memory\n");
        exit(1);
    }
    allocati = dimbloc;


    /* apre il file */
    fd=fopen(fileName, "r");
    if( fd==NULL ) {
        perror("Error, data file not found");
        exit(1);
    }

    /* leggo il file */
    while(fgets(buf, 200, fd)!=NULL) {
        usati += dimint;
        if(usati>allocati)
        {
            allocati += dimbloc;
            allData = (struct data *) realloc(allData, allocati);
            if(allData == NULL)
            {
                perror("Not enough memory\n");
                exit(1);
            }
            i++;
        }

        char *p=buf;
        while (*p) { // While there are more characters to process...
            if ( isdigit(*p) || ( (*p=='-'||*p=='+'||*p=='.') && isdigit(*(p+1)) )) {
                // Found a number
                double val = strtod(p, &p); // Read number
                printf("%f", val); // and print it.
            } else {
                // Otherwise, move on to the next character.
                p++;
            }
        }
    }


    /* chiude il file*/
    fclose(fd);

    return 0;
}