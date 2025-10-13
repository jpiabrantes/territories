#include <stdlib.h>
#include <math.h>

void shuffle(unsigned short* array, int n) {
    // Fisher-Yates shuffle
    for (short i = n - 1; i > 0; i--) {
        short j = rand() % (i + 1);
        short temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

// float normalise(float value, float min, float max) {
//     if (value > max) {
//         value = max;
//     } else if (value < min) {
//         value = min;
//     }
//     return (value - min) / (max - min) * 2 - 1;
// }

unsigned char float_to_byte(float value, float min, float max) {
    if (value > max) {
        value = max;
    } else if (value < min) {
        value = min;
    }
    return (unsigned char)(roundf((value - min) / (max - min) * 255.0f));
}