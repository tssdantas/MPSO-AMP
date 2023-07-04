#ifndef RANDOM_H
#define RANDOM_H

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <functional>
#include <limits>
#include <algorithm>
#include <random>

class Random
{
    public:
        Random() { }
        void SeedInput();
        double dRN(double, double);
        double iRN(int, int);
        std::mt19937 twisterengine;
};

#endif 