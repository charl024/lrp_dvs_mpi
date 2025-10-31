#ifndef HATS_HPP
#define HATS_HPP

#include <cstdint>
#include <vector>

typedef struct {
    uint32_t x;      // pixel x coordinate
    uint32_t y;      // pixel y coordinate
    uint32_t pol;    // polarity: 0 (OFF), 1 (ON)
    double t;        // timestamp
} Event_t;

// function to extract HATS features from a batch of events
std::vector<double> hats_processing(Event_t *input, int32_t size, int32_t width, int32_t height, int32_t cell_dim, int32_t rho, double delta, double tau);

#endif
