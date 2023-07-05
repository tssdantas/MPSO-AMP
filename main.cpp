#include <iomanip>
#include <iostream>
#include <vector>
#include <math.h>
#include <omp.h>
#include <vector>
#include "Random.h"
#include "inputData.h"
#include "MPSOAMP.h"

#define PI 3.14159265

// Rastrigin´s objective function
double objfun_rastrigin(std::vector<double> &lX)
{
    // // This code re-interprets void *my_func_data as pointer of the correct type during runtime:
    // static double wrap(const std::vector<double> &x, std::vector<double> &grad, void *data) {
    //     return (*reinterpret_cast<MyFunction*>(data))(x, grad); }

    double lx_sumterm = 0.00; double Y = 0.0;
    for (unsigned int j = 0; j < lX.size(); j++)
    {
        lx_sumterm += (pow(lX[j],2)) - 10*cos(2*PI*lX[j]);
    }
    // Y = data->nRast*data->nr_independent_variables + lx_sumterm;
    Y = (10)*2.0 + lx_sumterm;
    return Y;
}


int main ()
{
    string outputfilename = "Output.txt";
    InputData inputObj;
    MPsoAmp optObj(inputObj);
    optObj.OptMain(objfun_rastrigin, outputfilename, inputObj);
}