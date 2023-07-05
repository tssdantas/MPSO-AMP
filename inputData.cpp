#include <iostream>
#include <vector>
#include "inputData.h"
#include "Random.h"

using namespace std;

// CLASS::CONSTRUCTOR
InputData::InputData()  
{
    particles = 20;
    iterations = 30;
    use_random_seed = 0;
    seed = 1747814297812;
    nr_independent_variables = 2;
    xMin = -10.0; xMax = 10.0;
}