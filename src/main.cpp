#include "hats.hpp"
#include "hats_config.hpp"
#include "stdlib.h"
#include <vector>
#include <stdio.h>

int main(int argc, char** argv) 
{
    int N = atoi(argv[1]);
    Event_t* events = (Event_t *)malloc(N * N * sizeof(Event_t));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            events[i * N + j].x = j;
            events[i * N + j].y = i;
            events[i * N + j].pol = rand() % 2;
            events[i * N + j].t = i * N + j;
        }
    }

    std::vector<double> res = hats_processing(events, N * N, N, N, HATS_Config::CELL_DIM, HATS_Config::RHO, HATS_Config::DELTA, HATS_Config::TAU);


    // for (const auto& r: res) {
    //     printf("%0.3f\n", r);
    // }

    // printf("size: %lu\n", res.size());


    free(events);
    return 0;
}