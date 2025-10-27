#ifndef PGM_H
#define PGM_H

#include <stdio.h>
#include <stdlib.h>

class PGMImage {
public:
    int x_dim, y_dim;
    unsigned char *pixels;

    PGMImage(const char *filename) {
        FILE *f = fopen(filename, "rb");
        if (!f) {
            fprintf(stderr, "Error: no se pudo abrir el archivo %s\n", filename);
            exit(1);
        }

        char header[3];
        if (fscanf(f, "%2s", header) != 1 || header[0] != 'P' || header[1] != '5') {
            fprintf(stderr, "Error: formato PGM inválido (se esperaba P5)\n");
            fclose(f);
            exit(1);
        }

        // Ignora comentarios
        int c;
        while ((c = fgetc(f)) == '#') {
            while (fgetc(f) != '\n');
        }
        ungetc(c, f);

        int maxval;
        fscanf(f, "%d %d %d", &x_dim, &y_dim, &maxval);
        fgetc(f); // leer el salto de línea

        pixels = (unsigned char*)malloc(x_dim * y_dim);
        fread(pixels, 1, x_dim * y_dim, f);
        fclose(f);
    }

    ~PGMImage() {
        free(pixels);
    }
};

#endif
