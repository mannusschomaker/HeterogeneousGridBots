#pragma once
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


using namespace std;
//
//const int nSize = []() {
//    // Read the value from a file
//    int value;
//    std::ifstream inputFile("nSize.txt");
//    if (inputFile >> value) {
//        inputFile.close();
//    }
//    else {
//        std::cout << "Failed to read nSize from file." << std::endl;
//    }
//    return value;
//}();
//
//const double t_end_sim = []() {
//    // Read the value from a file
//    int value;
//    std::ifstream inputFile("t_end_sim.txt");
//    if (inputFile >> value) {
//        inputFile.close();
//    }
//    else {
//        std::cout << "Failed to read t end from file." << std::endl;
//    }
//    return value;
//}();
//
//const double data_step = []() {
//    // Read the value from a file
//    int value;
//    std::ifstream inputFile("data_step.txt");
//    if (inputFile >> value) {
//        inputFile.close();
//    }
//    else {
//        std::cout << "Failed to read step from file." << std::endl;
//    }
//    return value;
//}();
//
//const double nSim = []() {
//    // Read the value from a file
//    int value;
//    std::ifstream inputFile("nSim.txt");
//    if (inputFile >> value) {
//        inputFile.close();
//    }
//    else {
//        std::cout << "Failed to read nSim from file." << std::endl;
//    }
//    return value;
//}();
//
//
//
//
//const double cube_length = 0.05;
//const double spring_k = 1000.0;
//const double standard_mass = 0.075;
//const double A = 1;
//
//const double f = 0.85;
//const double alpha = 0.33;
//const double t_cycle = 2;
//const double t_motion = 1 / f;
//const double t_fixed = t_cycle - t_motion;
//const double d_1 = (t_cycle * alpha) - t_motion / 2;
////const double delta_l = 0.0035;
//const double delta_l = 0.007;
//const double omega = 2 * 3.14159 * f;
//const double ac[] = { A,f,t_cycle,t_motion,t_fixed,d_1 };
//
//const double Fcoulomb = 0.172;
//const double Fstatic = 0;
//const double Fbrk = Fcoulomb + Fstatic;
//const double F_constants[] = { Fbrk, Fcoulomb, Fstatic };
//const double Vbrk = 0.01;
//const double Vcoul = Vbrk / 10;
//const double Vst = Vbrk * sqrt(2);
//const double B = 2;
//const double V_constants[] = { Vbrk, Vcoul, Vst, B };
//
//const double epsilon = 5e-7;
//
//const int sSize = nSize - 1;
//const double spring_constants = spring_k;
//const double node_masses = standard_mass;
//const double rest_lengths = 0.05;

void eventFunction(double t, state_type y);