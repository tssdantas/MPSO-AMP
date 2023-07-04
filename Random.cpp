#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <functional>
#include <limits>
#include <algorithm>
#include <chrono>
#include <random>
#include "Random.h"

using namespace std;


void Random::SeedInput()
{
    // Mersenne Twiester Engine
    // Generates a seed with high resolution clock
    long long longseed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    longseed = longseed * (00000000.1);
    
    // 2. Initializes the random number generator with a twister engine object of the class MT19937
    twisterengine.seed(longseed);
}

double Random::dRN(double min, double max)
{
    // Generates an uniform distribution in range of [double min, double max]. 
    std::uniform_real_distribution<double> dist_real_uniform(min, max);
    // Returns random number within [min, max].
    return dist_real_uniform(twisterengine);
}

double Random::iRN(int min, int max)
{
    // Generates an uniform distribution in range of [int min, int max]. 
    std::uniform_int_distribution<int> dist_int_uniform(min, max - 1);
    // Returns random number within [min, max].
    return dist_int_uniform(twisterengine);
}