#ifndef INPUTDATA_H
#define INPUTDATA_H

#include <iostream>
#include <vector>
#include "InputData.h"


class InputData
{
    public:
        InputData();
        int particles;
        int use_random_seed;
        long long int seed;
        int iterations;
        int nr_independent_variables;
        double xMax, xMin;
        int nRast;
};

#endif 