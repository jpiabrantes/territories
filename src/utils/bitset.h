#ifndef BITSET_H
#define BITSET_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 

// BitSet structure
typedef struct {
    size_t max_num;   // maximum value representable (exclusive)
    size_t words;     // number of 64-bit words
    uint64_t *data;   // pointer to words (heap allocated)
} BitSet;

// Create a bitset for values 0..max_num (exclusive)
// Returns NULL on allocation failure.
BitSet *bitset_create(size_t max_num) {
    BitSet *bs = malloc(sizeof(BitSet));
    if (!bs)
      return NULL;
    bs->max_num = max_num;
    // number of bits needed = max_num
    bs->words = (max_num + 63) / 64; // ceil((max_num)/64)
    bs->data = calloc(bs->words, sizeof(uint64_t));
    if (!bs->data) {
      free(bs);
      return NULL;
    }
    return bs;
  }


static inline void bitset_clear(BitSet *bs) {
    if (!bs) return;
    memset(bs->data, 0, bs->words * sizeof(uint64_t));
}

// Free a bitset
void bitset_free(BitSet *bs) {
    if (!bs)
      return;
    free(bs->data);
    free(bs);
  }

// Add x to the bitset
static inline void bitset_add(BitSet *bs, size_t x) {
    if (x >= bs->max_num) return;
    size_t wi = x >> 6;
    unsigned off = x & 63;
    bs->data[wi] |= (1ULL << off);
}

// Remove x from the bitset
static inline void bitset_remove(BitSet *bs, size_t x) {
    if (x >= bs->max_num) return;
    size_t wi = x >> 6;
    unsigned off = x & 63;
    bs->data[wi] &= ~(1ULL << off);
}

// Check if x is in the bitset
static inline int bitset_contains(const BitSet *bs, size_t x) {
    if (x >= bs->max_num) return 0;
    size_t wi = x >> 6;
    unsigned off = x & 63;
    return (bs->data[wi] >> off) & 1U;
}

int bitset_update_members(const BitSet *bs, unsigned short* members) {
    int n_members = 0;
    for (size_t wi = 0; wi < bs->words; ++wi) {
        uint64_t word = bs->data[wi];
        while (word) {
            // find lowest set bit
            unsigned bit = __builtin_ctzll(word);
            size_t index = wi * 64 + (size_t)bit;
            members[n_members++] = (unsigned short)index;
            word &= word - 1; // clear the bit we just processed
        }
    }
    return n_members;
}

#endif // BITSET_H