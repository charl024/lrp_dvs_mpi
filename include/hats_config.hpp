#ifndef HATS_CONFIG_HPP
#define HATS_CONFIG_HPP
#pragma once

namespace HATS_Config {
    constexpr int WIDTH     = 640;
    constexpr int HEIGHT    = 480;
    constexpr int CELL_DIM  = 16; // or 8, important, small K = improved performance
    constexpr int RHO       = 2;  // or 2 pixels, minimal impact on accuracy
    constexpr double DELTA  = 10.0; // 5 ms, defines event memory event horizon (DELTA has to match dynamics of dataset)
    constexpr double TAU    = 5.0; // 10 ms, minimal impact on accuracy
}

#endif