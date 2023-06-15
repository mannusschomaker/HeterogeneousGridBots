// rk45_1d_sim.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cassert>
#include <chrono>
#include <cmath>
#include <math.h>
#include <stdlib.h>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <future>
#include <immintrin.h>  // Include the appropriate SIMD header (e.g., <immintrin.h> for AVX)
#include <boost/array.hpp>
#include <random>
#include <boost/numeric/odeint.hpp>
//#include "loadConstants.cpp"
#include "loadConstants.cpp"
//#include "eventFunction.h"
using namespace std;
using namespace boost::numeric::odeint;
typedef std::vector< double > state_type;

void fillWithPoisson(std::vector<double>& vec, double mean, double stddev) {
    // Create a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(mean, stddev);

    // Fill the vector with rounded random numbers from the normal distribution
    for (double& value : vec) {
        value = std::round(dist(gen));
    }
}

vector<double> get_y0(vector<double> nodes)
{
    vector<double> y_0(nodes.size(), 0.0);
    //vector<double> flat = nodes;
    y_0.insert(y_0.end(), nodes.begin(), nodes.end());
    return y_0;
}

const std::vector<double> createNodes() {


    std::vector<double> node_temp;
    for (int i = 0; i < nSize; i++) {
        node_temp.push_back(i * 0.05);  // Example values, modify as needed
    }

    return std::vector<double>(node_temp.begin(), node_temp.end());
}

std::vector<double> getRandomVector(int size) {

    std::random_device rd;
    std::mt19937 gen(rd()); // Seed the random number generator

    std::uniform_real_distribution<double> dist(0.0, 2.0);

    std::vector<double> random_vector(size);

    for (int i = 0; i < size; ++i) {
        random_vector[i] = dist(gen); // Generate a random double and assign it to the vector
    }

    return random_vector;
}

const std::vector<double> nodes = createNodes();

//vector<double> init_phase = getRandomVector(nSize - 1);


vector<double> y_init = get_y0(nodes);
const int ncnt = y_init.size();


// event functions
vector<double> t_unit(sSize, 0.0);
vector<double> phi_c = getRandomVector(sSize);
vector<double> phi_m = phi_c;
vector<int> event_c(sSize, 3);
vector<double> event_t_next(sSize, 0.0);
vector<double> measure(sSize, 0.0);
vector<double> score_c(sSize, 0.0);
vector<double> score_m(sSize, 0.0);
vector<double> ds(sSize, 0.1);

random_device rd;
mt19937 gen(rd()); // Seed the random number generator

uniform_real_distribution<double> ds_dist(-1.0, 1.0);

void eventFunction(double t, state_type y)
{
    for (size_t index = 0; index < t_unit.size(); index++)
    {
        if (event_c[index] == 0)
        {
            t_unit[index] = t - phi_m[index];
            if (event_t_next[index] < t)
            {
                event_c[index] = 1;
                // Adapt phase for phase transition
                event_t_next[index] = event_t_next[index] + t_cycle;
            }
        }
        else
        {
            t_unit[index] = t - phi_c[index];
        }
        
        if (event_c[index] == 1)
        {
            if (event_t_next[index] <= t)
            {
                // Make first measurement
                //measure[index] = (y[nSize + index] + y[nSize + index + 1])/2;
                measure[index] = (y[nSize] + y[nSize + 1]) / 2;
                //cout << measure[index] << " _ _" << index << endl;
                event_c[index] = 2;
                event_t_next[index] = event_t_next[index] + (t_cycle*2);
            }
        }

        if (event_c[index] == 2)
        {
            if (event_t_next[index] <= t)
            {

                //event_data[index] = phi_c[index] % 2;

                // Calculate the score for the block
                //score_c[index] =  ((y[nSize + index] + y[nSize + index + 1]) / 2) - measure[index];
                score_c[index] = ((y[nSize] + y[nSize + 1])/2) - measure[index];


                if (score_c[index] >= score_m[index]) {
                    phi_m[index] = phi_c[index];
                }
                score_m[index] = score_c[index];
                

                
                // Overwrite the old phases with the new phase
                phi_c[index] = double(phi_m[index] + (ds_dist(gen) * ds[index]));
                // Update the next event to be the phase's transition
                event_c[index] = 0;
                event_t_next[index] = event_t_next[index] + (t_cycle * 0.9);
                //cout << event_t_next[index] << endl;
            }
        }

        // Start actuation only after phase has started
        if (event_c[index] == 3)
        {
            t_unit[0] = 0;
            //phi_c[0] = 0;
            if (phi_c[0] <= t)
            {
                event_c[index] = 1;
                event_t_next[index] = phi_c[0];
            }
        }
    }

}




//
// integration-related functions
double actuation_length(double time)
{
    double t = fmod(time, t_cycle) - t_motion / 4;

    if (t < t_motion / 4)  // extension rise
        return ((sin(omega * t) + 1.0) * delta_l) + rest_lengths;
    else if (t < t_motion / 4 + d_1)  // extension plateau
        return ((sin(omega * t_motion / 4) + 1.0) * delta_l) + rest_lengths;
    else if (t < 3.0 / 4 * t_motion + d_1)  // extension fall
        return ((sin(omega * (t - d_1)) + 1.0) * delta_l) + rest_lengths;
    else if (t < 3.0 / 4 * t_motion + t_fixed)  // extension low plateau (contracted)
        return ((sin(omega * t_motion * 3.0 / 4) + 1.0) * delta_l) + rest_lengths;
    else if (t > t_motion * 3.0 / 4 + t_fixed)  // extension rise back to zero
        return ((sin(omega * (t - t_fixed)) + 1.0) * delta_l) + rest_lengths;
    else
        exit(1);  // impossible
}
//
double friction_term(double l)
{
    return -(2.3316439 * 0 + 0.172 * exp(-l * l / (Vbrk * Vbrk)) * (l / Vbrk)
        + Vcoul * tanh(l / Vcoul) + l * B);
}

class yprime_class {
    double m_gam;
    std::vector<double> phi;

public:
    yprime_class(double gam, const std::vector<double>& phi) : m_gam(gam), phi(phi) { }


    void operator() (const state_type& y, state_type& ydot, double t)
    {

        eventFunction(t, y);

        vector<double> acc(nSize, 0.0);  // force (acceleration in last loop) imparted on each node (component-wise);
        vector<double> l0(sSize, 0.0);
        for (int i = 0; i < sSize; i++) {  // actuate (new l0 = rest len + time-dependent)
            //l0[i] = actuation_length(t);
            //l0[i] = (cube_length + (sin(t)* cube_length*0.1));
            l0[i] = actuation_length(t_unit[i]);
        }
        //cout << endl;s
        for (int i = 0; i < sSize; i++) {  // Hooke's law
            //cout << phi_c[i] << ' _ ';

            int idx1 = nSize + i;
            int idx2 = nSize + i + 1;
            double yDiffX = y[idx2] - y[idx1];
            double lMinusL0 = yDiffX - l0[i];
            double factor = spring_constants * lMinusL0;

            double ax = factor * yDiffX;
            acc[i] += ax;
            acc[i + 1] -= ax;
        }

        for (int i = 0; i < nSize; i++) {  // friction
            double v = y[i];
            if (v == 0) v = 1;
            double frictionTermX = friction_term(v);
            acc[i] += frictionTermX;
        }
        //cout << endl;

        // y0 - vel, pos    y - [vel, pos]   ydot - [acc, vel]    never read ydot
        int ySize = nSize;
        for (int i = 0; i < ySize; i++) {

            ydot[i] = acc[i] / 0.075;
            ydot[i + nSize] = y[i];
        }


    }

};




// Define the observer class
struct FileObserver
{

    ofstream& file;
    size_t count_;

    FileObserver(ofstream& outputFile) : file(outputFile), count_(0) {}

    
    void operator()(const state_type& y, double t)
    {
        if (t > (float(count_)*data_step)) {
            // Write the integrated data to the file
            file << t << ",";
            for (const auto& value : y)
                file << value << ",";
            for (const auto& value : phi_c)
                file << value << ",";
            file << "\n";
            count_++;
        }
    }
};

int main(int argc, char** argv)
{


    const int numIterations = nSim;  // Number of iterations
    const std::string baseFileName = "poisson_mean_1_std_0.5";  // Base file name

    // Loop for each iteration
    for (int i = 1; i <= numIterations; ++i)
    {

        std::string fileName = baseFileName + std::to_string(i) + ".csv";

        // Open the output file
        std::ofstream outputFile(fileName);
        if (!outputFile.is_open())
        {
            std::cout << "Failed to open output file: " << fileName << std::endl;
            return 1;
        }

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        // init state array
        state_type y(nSize*2);
        cout << y_init.size();

        copy(y_init.begin(), y_init.end(), y.begin());

        yprime_class test(0.15, { 0 });
        //t_unit.assign(sSize, 0.0);
        phi_c = getRandomVector(sSize);
        phi_m = phi_c;
        //fill(event_c.begin(), event_c.end(), 3);
        event_c.assign(sSize, 3);
        event_t_next.assign(sSize, 0.0);
        measure.assign(sSize, 0.0);
        score_c.assign(sSize, 0.0);
        score_m.assign(sSize, 0.0);
        ds.assign(sSize, 0.05);

        //double mean = 0.1;
        //double stddev = 0.0001;

        //fillWithPoisson(ds, mean, stddev);

        //std::ifstream inputFile("ds_values.txt");
        //if (!inputFile.is_open()) {
        //    std::cout << "Failed to open the file." << std::endl;
        //    return 1;
        //}

        //for (int i = 0; i < nSize - 1; ++i) {
        //    inputFile >> ds[i];
        //}

        //inputFile.close();


        // intergrate with an adaptive step size
        typedef runge_kutta_cash_karp54<state_type> error_stepper_type;
        typedef controlled_runge_kutta<error_stepper_type> controlled_stepper_type;
        integrate_adaptive(make_controlled(1.0e-5, 1.0e-5, error_stepper_type()),
            test, y, 0.0, t_end_sim, 0.1, FileObserver(outputFile));

        // print integration time
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time taken s = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << std::endl;

        // Close the output file
        outputFile.close();
    }

    return 0;

}