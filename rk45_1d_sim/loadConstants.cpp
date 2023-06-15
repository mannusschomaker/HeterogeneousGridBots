#include <fstream>
#include <iostream>
#include <vector>
#include <random>



using namespace std;

const int nSize = []() {
    // Read the value from a file
    int value;
    std::ifstream inputFile("nSize.txt");
    if (inputFile >> value) {
        inputFile.close();
    }
    else {
        std::cout << "Failed to read nSize from file." << std::endl;
    }
    return value;
}();

const double t_end_sim = []() {
    // Read the value from a file
    int value;
    std::ifstream inputFile("t_end_sim.txt");
    if (inputFile >> value) {
        inputFile.close();
    }
    else {
        std::cout << "Failed to read t end from file." << std::endl;
    }
    return value;
}();

const double data_step = []() {
    // Read the value from a file
    int value;
    std::ifstream inputFile("data_step.txt");
    if (inputFile >> value) {
        inputFile.close();
    }
    else {
        std::cout << "Failed to read step from file." << std::endl;
    }
    return value;
}();

const double nSim = []() {
    // Read the value from a file
    int value;
    std::ifstream inputFile("nSim.txt");
    if (inputFile >> value) {
        inputFile.close();
    }
    else {
        std::cout << "Failed to read nSim from file." << std::endl;
    }
    return value;
}();




const double cube_length = 0.05;
const double spring_k = 1000.0;
const double standard_mass = 0.075;
const double A = 1;

const double f = 0.85;
const double alpha = 0.33;
const double t_cycle = 2;
const double t_motion = 1 / f;
const double t_fixed = t_cycle - t_motion;
const double d_1 = (t_cycle * alpha) - t_motion / 2;
//const double delta_l = 0.0035;
const double delta_l = 0.007;
const double omega = 2 * 3.14159 * f;
const double ac[] = { A,f,t_cycle,t_motion,t_fixed,d_1 };

const double Fcoulomb = 0.172;
const double Fstatic = 0;
const double Fbrk = Fcoulomb + Fstatic;
const double F_constants[] = { Fbrk, Fcoulomb, Fstatic };
const double Vbrk = 0.01;
const double Vcoul = Vbrk / 10;
const double Vst = Vbrk * sqrt(2);
const double B = 2;
const double V_constants[] = { Vbrk, Vcoul, Vst, B };

const double epsilon = 5e-7;

const int sSize = nSize - 1;
const double spring_constants = spring_k;
const double node_masses = standard_mass;
const double rest_lengths = 0.05;

//void eventFunction(double t, state_type y)
//{
//    for (size_t index = 0; index < t_unit.size(); index++)
//    {
//        if (event_c[index] == 0)
//        {
//            t_unit[index] = t - phi_m[index];
//            if (event_t_next[index] < t)
//            {
//                event_c[index] = 1;
//                // Adapt phase for phase transition
//                event_t_next[index] = event_t_next[index] + t_cycle;
//            }
//        }
//        else
//        {
//            t_unit[index] = t - phi_c[index];
//        }
//
//        if (event_c[index] == 1)
//        {
//            if (event_t_next[index] <= t)
//            {
//                // Make first measurement
//                //measure[index] = (y[nSize + index] + y[nSize + index + 1])/2;
//                measure[index] = (y[nSize] + y[nSize + 1]) / 2;
//                //cout << measure[index] << " _ _" << index << endl;
//                event_c[index] = 2;
//                event_t_next[index] = event_t_next[index] + (t_cycle * 2);
//            }
//        }
//
//        if (event_c[index] == 2)
//        {
//            if (event_t_next[index] <= t)
//            {
//
//                //event_data[index] = phi_c[index] % 2;
//
//                // Calculate the score for the block
//                //score_c[index] =  ((y[nSize + index] + y[nSize + index + 1]) / 2) - measure[index];
//                score_c[index] = ((y[nSize] + y[nSize + 1]) / 2) - measure[index];
//
//
//                if (score_c[index] >= score_m[index]) {
//                    phi_m[index] = phi_c[index];
//                }
//                score_m[index] = score_c[index];
//
//
//
//                // Overwrite the old phases with the new phase
//                phi_c[index] = double(phi_m[index] + (ds_dist(gen) * ds[index]));
//                // Update the next event to be the phase's transition
//                event_c[index] = 0;
//                event_t_next[index] = event_t_next[index] + (t_cycle * 0.9);
//                //cout << event_t_next[index] << endl;
//            }
//        }
//
//        // Start actuation only after phase has started
//        if (event_c[index] == 3)
//        {
//            t_unit[0] = 0;
//            //phi_c[0] = 0;
//            if (phi_c[0] <= t)
//            {
//                event_c[index] = 1;
//                event_t_next[index] = phi_c[0];
//            }
//        }
//    }
//
//}