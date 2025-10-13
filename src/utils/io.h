#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "paths.h"

bool* read_is_soil(int width, int height) {
    bool* is_soil = (bool*)malloc(width * height * sizeof(bool));
    char filename[64];
    snprintf(filename, sizeof(filename), "resources/is_soil_%d_%d.bin", width, height);
    FILE* file = fopen(filename, "rb");    
    if (!file) {
        fprintf(stderr, "Failed to open file %s\n", filename);
        return NULL;
    }
    size_t read_size = fread(is_soil, sizeof(bool), width * height, file);
    if ((int)read_size != width * height) {
        fprintf(stderr, "Wrong dimensions when reading file %s\n", filename);
        return NULL;
    }
    fclose(file);
    return is_soil;
}