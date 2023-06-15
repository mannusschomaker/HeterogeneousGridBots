
#include <iostream>
#include <fstream>
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


#include "LSODA.h"
#include "helper.h"

using namespace std;


// physical constants
const double cube_length = 0.046;
const double spring_k = 500;
const double standard_mass = 0.075;
const double A = 1;

const double f = 0.85;
const double alpha = 0.33;
const double t_cycle = 2;
const double t_motion = 1 / f;
const double t_fixed = t_cycle - t_motion;
const double d_1 = (t_cycle * alpha) - t_motion / 2;
const double delta_l = 0.0035;
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

// precompute the model structure (many redundancies, but doesn't matter because only generated once)
vector<vector<double>> get_model_positions(vector<vector<int>> mat)
{
    // returns pairs of coordinates for all the nodes
    vector<vector<double>> nodes;
    for (int i = 0; i < (int)mat.size(); i++) {
        for (int j = 0; j < (int)mat[0].size(); j++) {
            if (mat[i][j] == 1) {
                vector<double> node1 = { cube_length * static_cast<double>(i), cube_length * static_cast<double>(j) };
                vector<double> node2 = { cube_length * (static_cast<double>(i) + 1), cube_length * static_cast<double>(j) };
                vector<double> node3 = { cube_length * static_cast<double>(i), cube_length * (static_cast<double>(j) + 1) };
                vector<double> node4 = { cube_length * (static_cast<double>(i) + 1), cube_length * (static_cast<double>(j) + 1) };
                if (std::find(nodes.begin(), nodes.end(), node1) == nodes.end()) nodes.push_back(node1);
                if (std::find(nodes.begin(), nodes.end(), node2) == nodes.end()) nodes.push_back(node2);
                if (std::find(nodes.begin(), nodes.end(), node3) == nodes.end()) nodes.push_back(node3);
                if (std::find(nodes.begin(), nodes.end(), node4) == nodes.end()) nodes.push_back(node4);
            }
        }
    }
    return nodes;
}
vector<vector<double>> centre_nodes(vector<vector<double>> node_positions)
{
    vector<vector<double>> pos;
    double avgx = 0.0, avgy = 0.0;
    for (vector<double> v : node_positions) {
        avgx += v[0]; avgy += v[1];
    }
    avgx /= (double)node_positions.size(); avgy /= (double)node_positions.size();
    for (vector<double> v : node_positions) {
        pos.push_back(vector<double> {v[0] - avgx, v[1] - avgy});
    }
    return pos;
}

pair<vector<vector<int>>, vector<int>> get_model_springs(vector<vector<int>> mat, vector<vector<double>> nodes)
{
    // returns pairs of indices in nodelist (both u-v and v-u) for all the springs + list of indices of actuated (diagonal springs)
    // NOTE: diagonal springs for block n (always counting top->bot,left->right) have indices 4n,4n+1,4n+2,4n+3 in actuators list by design
    vector<vector<int>> springs;
    vector<int> actuators;
    for (int i = 0; i < (int)mat.size(); i++) {
        for (int j = 0; j < (int)mat[0].size(); j++) {
            if (mat[i][j] == 1) {
                auto it1 = std::find(nodes.begin(), nodes.end(), vector<double> { cube_length* static_cast<double>(i), cube_length* static_cast<double>(j) });
                auto it2 = std::find(nodes.begin(), nodes.end(), vector<double> { cube_length* (static_cast<double>(i) + 1.), cube_length* static_cast<double>(j) });
                auto it3 = std::find(nodes.begin(), nodes.end(), vector<double> { cube_length* static_cast<double>(i), cube_length* (static_cast<double>(j) + 1.) });
                auto it4 = std::find(nodes.begin(), nodes.end(), vector<double> { cube_length* (static_cast<double>(i) + 1.), cube_length* (static_cast<double>(j) + 1.) });
                springs.push_back(vector<int> { static_cast<int>(it1 - nodes.begin()), static_cast<int>(it4 - nodes.begin()) });
                actuators.push_back(static_cast<int>(springs.size() - 1));
                springs.push_back(vector<int> { static_cast<int>(it2 - nodes.begin()), static_cast<int>(it3 - nodes.begin()) });
                actuators.push_back(static_cast<int>(springs.size() - 1));
                springs.push_back(vector<int> { static_cast<int>(it1 - nodes.begin()), static_cast<int>(it2 - nodes.begin()) });
                springs.push_back(vector<int> { static_cast<int>(it1 - nodes.begin()), static_cast<int>(it3 - nodes.begin()) });
                springs.push_back(vector<int> { static_cast<int>(it3 - nodes.begin()), static_cast<int>(it4 - nodes.begin()) });
                springs.push_back(vector<int> { static_cast<int>(it2 - nodes.begin()), static_cast<int>(it4 - nodes.begin()) });
            }
        }
    }
    return make_pair(springs, actuators);
}

vector<double> get_spring_constants(vector<vector<int>> mat, vector<vector<int>> springs, vector<vector<double>> nodes)
{
    // list of spring constants (same order as spring list)
    vector<double> spring_constants(springs.size(), spring_k);
    for (int i = 0; i < (int)mat.size(); i++) {
        for (int j = 0; j < (int)mat[0].size(); j++) {
            auto it2 = std::find(nodes.begin(), nodes.end(), vector<double> {cube_length* (static_cast<double>(i) + 1.), cube_length* static_cast<double>(j)});
            auto it3 = std::find(nodes.begin(), nodes.end(), vector<double> {cube_length* static_cast<double>(i), cube_length* (static_cast<double>(j) + 1.)});
            auto it4 = std::find(nodes.begin(), nodes.end(), vector<double> {cube_length* (static_cast<double>(i) + 1.), cube_length* (static_cast<double>(j) + 1.)});
            if ((mat[i][j] == 1) && (i < (int)mat.size() - 1)) {
                if (mat[i + 1][j] == 1) {
                    auto springBottom = std::find(springs.begin(), springs.end(), vector<int> {(int)(it2 - nodes.begin()), (int)(it4 - nodes.begin())});
                    spring_constants[(int)(springBottom - springs.begin())] = 2 * spring_k;
                }
            }
            if ((mat[i][j] == 1) && (j < (int)mat[0].size() - 1)) {
                if (mat[i][j + 1] == 1) {
                    auto springRight = std::find(springs.begin(), springs.end(), vector<int> {(int)(it3 - nodes.begin()), (int)(it4 - nodes.begin())});
                    spring_constants[(int)(springRight - springs.begin())] = 2 * spring_k;
                }
            }
        }
    }
    return spring_constants;
}

vector<double> get_masses(vector<vector<int>> springs, vector<vector<double>> nodes)
{
    vector<int> deg(nodes.size(), 0);
    for (vector<int> spring : springs) {
        deg[spring[0]] += 1;
        deg[spring[1]] += 1;
    }
    vector<double> masses(nodes.size(), 0);
    for (int i = 0; i < (int)nodes.size(); i++) {
        masses[i] = deg[i] * standard_mass;
    }
    return masses;
}

double d(vector<double> x, vector<double> y)
{
    // Euclidean distance, length must be 2, datatype vector only for convenience 
    return sqrt(pow(x[0] - y[0], 2) + pow(x[1] - y[1], 2));
}

vector<double> get_rest_lengths(vector<vector<int>> springs, vector<vector<double>> nodes)
{
    vector<double> lengths;
    for (vector<int> spring : springs) {
        lengths.push_back(d(nodes[spring[0]], nodes[spring[1]]));
    }
    return lengths;
}

vector<double> flatten_nodes(vector<vector<double>> nodes)
{
    vector<double> res(2 * nodes.size(), 0);
    for (int i = 0; i < (int)nodes.size(); i++) {
        res[2 * i] = nodes[i][0]; res[2 * i + 1] = nodes[i][1];
    }
    return res;
}

vector<double> get_y0(vector<vector<double>> nodes)
{
    vector<double> y_0(2 * nodes.size(), 0.0);
    vector<double> flat = flatten_nodes(centre_nodes(nodes));
    y_0.insert(y_0.end(), flat.begin(), flat.end());
    return y_0;
}

const vector<vector<int>> shape = {{1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1}};
const vector<vector<double>> nodes = get_model_positions(shape);  // {{-0.05,0},{0.05,0}}; //
const vector<vector<int>> springs = get_model_springs(shape, nodes).first; // {{0,1}};
const vector<int> actuators = get_model_springs(shape, nodes).second; // {0.};
const vector<double> spring_constants = get_spring_constants(shape, springs, nodes);//{100.0};
const vector<double> node_masses = get_masses(springs, nodes);
const vector<double> rest_lengths = get_rest_lengths(springs, nodes);
vector<double> y_init = get_y0(nodes);
const int ncnt = y_init.size();
const int blockcount = 3;
const int nSize = nodes.size();
const int sSize = springs.size();


array<double, 2> rotate(double x, double y, double theta) {
    return array<double, 2> {x* cos(theta) - y * sin(theta), x* sin(theta) + y * cos(theta)};
}

array<double, 2> com(vector<double> pos)
{
    array<double, 2> c{ 0,0 };
    for (int j = 0; j < (int)pos.size() / 2; j++) {
        c[0] += pos[2 * j]; c[1] += pos[2 * j + 1];
    }
    c[0] /= ((double)pos.size() / 2.); c[1] /= ((double)pos.size() / 2.);
    return c;
}

double angVec(double x1in, double y1in, double x2in, double y2in)
{
    if (((abs(x1in) < epsilon) && (abs(y1in) < epsilon)) || ((abs(x2in) < epsilon) && (abs(y2in) < epsilon))) {
        cout << "err: null vector: (" << x1in << "," << y1in << "), (" << x2in << "," << y2in << ")" << endl;
        exit(1);
    }
    // normalise and get angle by dot product
    double l1 = sqrt(pow(x1in, 2.) + pow(y1in, 2.)), l2 = sqrt(pow(x2in, 2.) + pow(y2in, 2.));
    double x1 = x1in / l1, y1 = y1in / l1, x2 = x2in / l2, y2 = y2in / l2;
    double prod = x1 * x2 + y1 * y2;
    if (abs(prod - 1.) < epsilon) return 0;
    if (abs(prod - -1.) < epsilon) return 3.14159;
    double theta = acos(prod);
    array<double, 2> rot = rotate(x1, y1, theta);
    if ((abs(rot[0] - x2) < epsilon) && (abs(rot[1] - y2) < epsilon)) return theta;
    rot = rotate(x1, y1, -theta);
    if ((abs(rot[0] - x2) < epsilon) && (abs(rot[1] - y2) < epsilon)) return -theta;
    // check if error (e.g. precision too low), mathematically impossible  
    cout << "ang err: " << x1 << "," << y1 << " ang " << x2 << "," << y2 << " theta=" << theta << endl;
    exit(1);
}

double angLst(vector<double> a, vector<double> b)
{
    // assumes a and b have the same size
    double avg = 0; array<double, 2> com1 = com(a), com2 = com(b);
    for (int j = 0; j < (int)a.size() / 2; j++) {
        avg += angVec(a[2 * j] - com1[0], a[2 * j + 1] - com1[1], b[2 * j] - com2[0], b[2 * j + 1] - com2[1]);
    }
    return avg / ((double)a.size() / 2.);
}


// integration-related functions
double actuation_length(double time)
{
    double t = fmod(time, t_cycle) - t_motion / 4;

    if (t < t_motion / 4)  // extension rise
        return (sin(omega * t) + 1.0) * delta_l;
    else if (t < t_motion / 4 + d_1)  // extension plateau
        return (sin(omega * t_motion / 4) + 1.0) * delta_l;
    else if (t < 3.0 / 4 * t_motion + d_1)  // extension fall
        return (sin(omega * (t - d_1)) + 1.0) * delta_l;
    else if (t < 3.0 / 4 * t_motion + t_fixed)  // extension low plateau (contracted)
        return (sin(omega * t_motion * 3.0 / 4) + 1.0) * delta_l;
    else if (t > t_motion * 3.0 / 4 + t_fixed)  // extension rise back to zero
        return (sin(omega * (t - t_fixed)) + 1.0) * delta_l;
    else
        exit(1);  // impossible
}

double friction_term(double x, double l)
{
    return -(2.3316439 * 0 + 0.172 * exp(-l * l / (Vbrk * Vbrk)) * (l / Vbrk)
        + Vcoul * tanh(l / Vcoul) + l * B) * (x / l);
}

void yprime(double t, double* y, double* ydot, void* phi)
{
    (void)phi;

    vector<double> acc(nSize * 2., 0.0);  // force (acceleration in last loop) imparted on each node (component-wise)
    vector<double> l0(rest_lengths);


    for (int i = 0; i < (int)actuators.size(); i++) {  // actuate (new l0 = rest len + time-dependent)
        double actuationLen = actuation_length(t - static_cast<double*>(phi)[div(i, 2).quot]);
        l0[actuators[i]] += actuationLen;
    }

    for (int i = 0; i < sSize; i++) {  // Hooke's law
        int idx1 = nSize * 2 + 2 * springs[i][0];
        int idx2 = nSize * 2 + 2 * springs[i][1];
        double yDiffX = y[idx1] - y[idx2];
        double yDiffY = y[idx1 + 1] - y[idx2 + 1];
        double lMinusL0 = sqrt(yDiffX * yDiffX + yDiffY * yDiffY) - l0[i];
        double factor = -spring_constants[i] * lMinusL0 / sqrt(yDiffX * yDiffX + yDiffY * yDiffY);
        double ax = factor * yDiffX;
        double ay = factor * yDiffY;
        acc[2 * springs[i][0]] += ax;
        acc[2 * springs[i][0] + 1] += ay;
        acc[2 * springs[i][1]] -= ax;
        acc[2 * springs[i][1] + 1] -= ay;
        // TODO: springs only once
    }

    for (int i = 0; i < nSize; i++) {  // friction
        int idx1 = 2 * i;
        int idx2 = idx1 + 1;
        double yValX = y[idx1];
        double yValY = y[idx2];
        double l = sqrt(yValX * yValX + yValY * yValY);
        if (l == 0) l = 1;
        double frictionTermX = friction_term(yValX, l);
        double frictionTermY = friction_term(yValY, l);
        acc[idx1] += frictionTermX;
        acc[idx2] += frictionTermY;
    }

    // y0 - vel, pos    y - [vel, pos]   ydot - [acc, vel]    never read ydot
    int ySize = nSize * 2;
    for (int i = 0; i < ySize; i++) {
        ydot[i + 2 * nSize] = y[i];
        ydot[i] = acc[i] / node_masses[div(i, 2).quot];
    }
}

vector<double> integrate(double phases[])
{
    double t = 0; vector<double> res, init, previousPos; int istate = 1; LSODA lsoda;
    init = y_init; previousPos = vector<double>(init.begin() + ncnt / 2, init.end());
    double t_max = 200., step_size = 1.; vector<double> comx, comy, ang; double angSum = 0;
    for (double i = 1.; i <= t_max; i += step_size) {
        lsoda.lsoda_update(yprime, y_init.size(), init, res, &t, i * step_size, &istate, phases, 1e-4, 1e-4);
        //if ((int)i % 10 == 0) {
            //vector<double> currentPos = vector<double>(res.begin() + ncnt / 2 + 1, res.end());
            //double angDiff = angLst(previousPos, currentPos);
            //ang.push_back(angDiff); angSum += angDiff;
            //array<double, 2> c1 = com(previousPos), c2 = com(currentPos); // can be optimised (redundant)
            //array<double, 2> corrected = rotate(c2[0] - c1[0], c2[1] - c1[1], -angSum);
            //comx.push_back(corrected[0]); comy.push_back(corrected[1]);
            /*cout << t << ",";
            for (int i = 0; i < (int)previousPos.size(); i++) cout << previousPos[i] << ",";
            cout << endl;
            cout << c2[0] << "," << c2[1] << "," << angLst(previousPos,currentPos) << endl;*/
            //previousPos = vector<double>(currentPos.begin(), currentPos.end());
        //}
        //init = vector<double>(res); res = vector<double>();
    }
    //vector<double> stats;
    //for (int i = 0; i < blockcount; i++) stats.push_back(phases[i]);
    //stats.push_back(accumulate(comx.begin() + 1, comx.end(), 0.) / (double)(comx.size() - 1));
    //stats.push_back(accumulate(comy.begin() + 1, comy.end(), 0.) / (double)(comy.size() - 1));
    //stats.push_back(accumulate(ang.begin() + 1, ang.end(), 0.) / (double)(ang.size() - 1));
    //return stats;
    return {};
}

//template <typename RAIter>
//int parallel_sum(RAIter beg, RAIter end)
//{
//    auto len = end - beg;
//    if (len < 1000) {
//        return std::accumulate(beg, end, 0);
//    }
//    RAIter mid = beg + len / 2;
//    auto handle = std::async(std::launch::async, parallel_sum<RAIter>, mid, end);
//    int sum = parallel_sum(beg, mid);
//    return sum + handle.get();
//}

void simple_map()
{
    double delta = 2.;
    vector<vector<double>> res;
    for (double x = 0.; x <= 2.; x += delta) {
        for (double y = 0.; y <= 2.; y += delta) {
            double phases[] = { 0,x,y,0,0,0,0,0, 0,x,y,0,0,0,0,0 ,0,x,y,0,0,0,0,0 , 0,x,y,0,0,0,0,0, 0,x,y,0,0,0,0,0, 0,x,y,0,0,0,0,0, 0,x,y,0,0,0,0,0, };
            integrate(phases);
        }
        cout << "x=" << x << endl;
    }
    cout << "write to file ... " << endl;
    ofstream myfile;
    myfile.open("test4.txt", ofstream::trunc);
    for (vector<double> el : res) {
        for (double d : el) {
            myfile << d << ",";
        }
        myfile << endl;
    }
    myfile.close();
    cout << "end" << endl;
}

int main(int argc, const char* argv[])
{
    (void)argc;
    (void)argv;

    /*cout << "nodes:";
    for (vector<double> n : nodes) cout << n[0] << "," << n[1] <<  "  ";
    cout << endl << "springs: ";
    for (vector<int> spring : springs) cout << spring[0] << "," << spring[1] << "  ";
    cout << endl << "actuators: ";
    for (int i : actuators) cout << i << " ";
    cout << endl << "spring constants: ";
    for (int x : spring_constants) cout << x << " ";
    cout << endl << "masses: ";
    for (double m : node_masses) cout << m << " ";*/

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    //vector<double> alg {-1,-1,1,1,-1,1,1,-1,0,1};
    //vector<double> blg {-1,0,3,0,1,2,1,-2,3,2};
    //double phases[] = {0.,1.,0.4};
    //assert((sizeof(phases)/sizeof(*phases) == actuators.size() / 2));
    //assert((sizeof(phases)/sizeof(*phases) == blockcount));
    //sintegrate(phases);
    simple_map();
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "Time taken s= " << chrono::duration_cast<chrono::seconds>(end - begin).count() << endl;
    (void)end;
    (void)begin;
    //cout << endl << "y0: ";
    //for (double x : y_init) cout << x << " ";
    //cout << endl; 

    //std::vector<int> v(10000, 1);
    //std::cout << "The sum is " << parallel_sum(v.begin(), v.end()) << '\n';

    return 0;
}