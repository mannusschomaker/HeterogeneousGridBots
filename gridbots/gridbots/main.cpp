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


#include "libsoda-cxx/LSODA.h"
#include "libsoda-cxx/helper.h"

using namespace std;


// physical constants
const double cube_length = 0.046;
const double spring_k = 500;
const double standard_mass = 0.075;
const double A = 1;

const double f = 0.85;
const double alpha = 0.33;
const double t_cycle = 2;
const double t_motion = 1/f;
const double t_fixed = t_cycle - t_motion;
const double d_1 = (t_cycle * alpha) - t_motion/2;
const double delta_l = 0.0035;
const double omega = 2 * M_PI * f;
const double ac[] = {A,f,t_cycle,t_motion,t_fixed,d_1};

const double Fcoulomb = 0.172;
const double Fstatic = 0;
const double Fbrk = Fcoulomb + Fstatic;
const double F_constants[] = {Fbrk, Fcoulomb, Fstatic};
const double Vbrk = 0.01;
const double Vcoul = Vbrk/10;
const double Vst = Vbrk * sqrt(2);
const double B = 2;  // should this be 4? 
const double V_constants[] = {Vbrk, Vcoul, Vst, B};

const double epsilon = 5e-7;

// precompute the model structure (many redundancies, but doesn't matter because only generated once)
vector<vector<double>> get_model_positions(vector<vector<int>> mat)
{
    // returns pairs of coordinates for all the nodes
    vector<vector<double>> nodes;
    for (int i = 0; i < (int)mat.size(); i++) {
        for (int j = 0; j < (int)mat[0].size(); j++) {
            if (mat[i][j] == 1) {
                vector<double> node1 = {cube_length * i, cube_length * j};
                vector<double> node2 = {cube_length * (i+1), cube_length * j};
                vector<double> node3 = {cube_length * i, cube_length * (j+1)};
                vector<double> node4 = {cube_length * (i+1), cube_length * (j+1)};
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
        pos.push_back(vector<double> {v[0]-avgx,v[1]-avgy});
    }
    return pos;
}

pair<vector<vector<int>>,vector<int>> get_model_springs(vector<vector<int>> mat, vector<vector<double>> nodes)
{
    // returns pairs of indices in nodelist (both u-v and v-u) for all the springs + list of indices of actuated (diagonal springs)
    // NOTE: diagonal springs for block n (always counting top->bot,left->right) have indices 4n,4n+1,4n+2,4n+3 in actuators list by design
    vector<vector<int>> springs;
    vector<int> actuators;
    for (int i = 0; i < (int)mat.size(); i++) {
        for (int j = 0; j < (int)mat[0].size(); j++) {
            if (mat[i][j] == 1) {
                auto it1 = std::find(nodes.begin(), nodes.end(), vector<double> {cube_length * i, cube_length * j});
                auto it2 = std::find(nodes.begin(), nodes.end(), vector<double> {cube_length * (i+1), cube_length * j});
                auto it3 = std::find(nodes.begin(), nodes.end(), vector<double> {cube_length * i, cube_length * (j+1)});
                auto it4 = std::find(nodes.begin(), nodes.end(), vector<double> {cube_length * (i+1), cube_length * (j+1)});
                springs.push_back(vector<int> {(int)(it1 - nodes.begin()), (int)(it4 - nodes.begin())});
                actuators.push_back(springs.size()-1);
                springs.push_back(vector<int> {(int)(it2 - nodes.begin()), (int)(it3 - nodes.begin())});
                actuators.push_back(springs.size()-1);
                springs.push_back(vector<int> {(int)(it1 - nodes.begin()), (int)(it2 - nodes.begin())});
                springs.push_back(vector<int> {(int)(it1 - nodes.begin()), (int)(it3 - nodes.begin())});
                springs.push_back(vector<int> {(int)(it3 - nodes.begin()), (int)(it4 - nodes.begin())});
                springs.push_back(vector<int> {(int)(it2 - nodes.begin()), (int)(it4 - nodes.begin())});
            }
        }
    }
    return std::make_pair(springs, actuators);
}

vector<double> get_spring_constants(vector<vector<int>> mat, vector<vector<int>> springs, vector<vector<double>> nodes)
{
    // list of spring constants (same order as spring list)
    vector<double> spring_constants(springs.size(), spring_k);
    for (int i = 0; i < (int)mat.size(); i++) {
        for (int j = 0; j < (int)mat[0].size(); j++) {
            auto it2 = std::find(nodes.begin(), nodes.end(), vector<double> {cube_length * (i+1), cube_length * j});
            auto it3 = std::find(nodes.begin(), nodes.end(), vector<double> {cube_length * i, cube_length * (j+1)});
            auto it4 = std::find(nodes.begin(), nodes.end(), vector<double> {cube_length * (i+1), cube_length * (j+1)});
            if ((mat[i][j] == 1) && (i < (int)mat.size()-1)) {
                if (mat[i+1][j] == 1) {
                    auto springBottom = std::find(springs.begin(), springs.end(), vector<int> {(int)(it2 - nodes.begin()), (int)(it4 - nodes.begin())});
                    spring_constants[(int)(springBottom - springs.begin())] = 2 * spring_k;
                }
            }
            if ((mat[i][j] == 1) && (j < (int)mat[0].size()-1)) {
                if (mat[i][j+1] == 1) {
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
    vector<int> deg(nodes.size(),0);
    for (vector<int> spring : springs) {
        deg[spring[0]] += 1;
        deg[spring[1]] += 1;
    }
    vector<double> masses(nodes.size(),0);
    for (int i = 0; i < (int)nodes.size(); i++) {
        masses[i] = deg[i] * standard_mass;
    }
    return masses;
}

double d(vector<double> x, vector<double> y)
{
    // Euclidean distance, length must be 2, datatype vector only for convenience
    return sqrt(pow(x[0]-y[0],2) + pow(x[1]-y[1],2));
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
    vector<double> res(2*nodes.size(),0);
    for (int i = 0; i < (int)nodes.size(); i++) {
        res[2*i] = nodes[i][0]; res[2*i + 1] = nodes[i][1];
    }
    return res;
}

vector<double> get_y0(vector<vector<double>> nodes)
{
    vector<double> y_0(2*nodes.size(),0.0);
    vector<double> flat = flatten_nodes(centre_nodes(nodes));
    y_0.insert(y_0.end(),flat.begin(),flat.end());
    return y_0;
}

const vector<vector<int>> shape = {{1,1},{1,0}};
const vector<vector<double>> nodes = get_model_positions(shape);  // {{-0.05,0},{0.05,0}}; //
const vector<vector<int>> springs = get_model_springs(shape,nodes).first; // {{0,1}};
const vector<int> actuators = get_model_springs(shape,nodes).second; // {0.};
const vector<double> spring_constants = get_spring_constants(shape, springs, nodes);//{100.0};
const vector<double> node_masses = get_masses(springs, nodes);
const vector<double> rest_lengths = get_rest_lengths(springs, nodes);
vector<double> y_init = get_y0(nodes);
const int ncnt = y_init.size();
const int blockcount = 3;


array<double,2> rotate(double x, double y, double theta) {
    return array<double,2> {x*cos(theta) - y*sin(theta), x*sin(theta) + y*cos(theta)};
}

array<double,2> com(vector<double> pos)
{
    array<double,2> c{0,0};
    for (int j = 0; j < (int)pos.size()/2; j++) {
        c[0] += pos[2*j]; c[1] += pos[2*j+1];
    }
    c[0] /= ((double)pos.size()/2.); c[1] /= ((double)pos.size()/2.);
    return c;
}

double angVec(double x1in, double y1in, double x2in, double y2in)
{
    if (((abs(x1in) < epsilon) && (abs(y1in) < epsilon)) || ((abs(x2in) < epsilon) && (abs(y2in) < epsilon))) {
        cout << "err: null vector: (" << x1in << "," << y1in << "), (" << x2in << "," << y2in << ")" << endl;
        exit(1);
    }
    // normalise and get angle by dot product
    double l1 = sqrt(pow(x1in,2.)+pow(y1in,2.)), l2 = sqrt(pow(x2in,2.)+pow(y2in,2.));
    double x1 = x1in/l1, y1 = y1in/l1, x2 = x2in/l2, y2 = y2in/l2;
    double prod = x1*x2 + y1*y2;
    if (abs(prod - 1.) < epsilon) return 0;
    if (abs(prod - -1.) < epsilon) return M_PI;
    double theta = acos(prod);
    array<double,2> rot = rotate(x1,y1,theta);
    if ((abs(rot[0]-x2) < epsilon) && (abs(rot[1]-y2) < epsilon)) return theta;
    rot = rotate(x1,y1,-theta);
    if ((abs(rot[0]-x2) < epsilon) && (abs(rot[1]-y2) < epsilon)) return -theta;
    // check if error (e.g. precision too low), mathematically impossible
    cout << "ang err: " << x1 << "," << y1 << " ang " << x2 << "," << y2 << " theta=" << theta << endl;
    exit(1);
}

double angLst(vector<double> a, vector<double> b)
{
    // assumes a and b have the same size
    double avg = 0; array<double,2> com1 = com(a), com2 = com(b);
    for (int j = 0; j < (int)a.size()/2; j++) {
        avg += angVec(a[2*j] - com1[0], a[2*j+1] - com1[1], b[2*j] - com2[0], b[2*j+1] - com2[1]);
    }
    return avg / ((double)a.size()/2.);
}


// integration-related functions
double actuation_length(double time)
{
    double t = fmod(time,t_cycle) - ac[3]/4;
    if (t < ac[3]/4.)  // extension rise
        return ac[0] * (sin(omega * t) + 1.) * delta_l;
    else if ((t >= ac[3]/4.) && (t < ac[3]/4. + ac[5]))  // extension plateau
        return ac[0] * (sin(omega * ac[3]/4.) + 1) * delta_l;
    else if ((t >= ac[3]/4. + ac[5]) && (t < 3./4.*ac[3] + ac[5]))  // extension fall
        return ac[0] * (sin(omega * (t - ac[5])) + 1.) * delta_l;
    else if ((t >= 3./4.*ac[3] + ac[5]) && (t < 3./4. * ac[3] + ac[4]))  // extension low plateau (contracted)
        return ac[0] * (sin(omega * ac[3] * 3./4.) + 1.) * delta_l;
    else if (t > ac[3]*3./4. + ac[4]) // extention rise back to zero
        return ac[0] * (sin(omega * (t - ac[4])) + 1.) * delta_l;
    else
        exit(1);  // impossible
}

double friction_term(double x, double l)
{
    return -(2.3316439 * F_constants[2] * exp(-pow(l/V_constants[2],2)) * (l / V_constants[2])
        +  F_constants[1] * tanh(l/V_constants[1])  +  l * V_constants[3]) * (x / l);
}

void yprime(double t, double* y, double* ydot, void* phi)
{
    vector<double> acc(nodes.size() * 2, 0.0);  // force (acceleration in last loop) imparted on each node (component-wise)
    vector<double> l0(rest_lengths);
    for (int i = 0; i < (int)actuators.size(); i++) {  // actuate (new l0 = rest len + time-dependent)
        l0[actuators[i]] += actuation_length(t - ((double*)phi)[div(i,2).quot]);
    }
    int ix1, iy1, ix2, iy2;  // flattened-out indices of the components of start/end node of a spring
    double l, ax, ay;
    for (int i = 0; i < (int)springs.size(); i++) {  // Hooke's law
        ix1 = 2*nodes.size() + 2*springs[i][0]; iy1 = ix1+1;
        ix2 = 2*nodes.size() + 2*springs[i][1]; iy2 = ix2+1;
        l = sqrt(pow(y[ix1]-y[ix2],2.0) + pow(y[iy1]-y[iy2],2.0));
        ax = -spring_constants[i] * (y[ix1]-y[ix2]) * (l-l0[i]) / l;
        ay = -spring_constants[i] * (y[iy1]-y[iy2]) * (l-l0[i]) / l;
        acc[2*springs[i][0]] += ax; acc[2*springs[i][0] + 1] += ay;
        acc[2*springs[i][1]] -= ax; acc[2*springs[i][1] + 1] -= ay;
    }
    for (int i = 0; i < (int)nodes.size(); i++) {  // friction
        ix1 = 2*i; iy1 = ix1 + 1;
        l = sqrt(pow(y[ix1],2.0) + pow(y[iy1],2.0));  if (l==0) l = 1;
        acc[2*i] += friction_term(y[ix1], l);
        acc[2*i+1] += friction_term(y[iy1], l);
    }
    // y0 - vel, pos    y - [vel, pos]   ydot - [acc, vel]    never read ydot
    for (int i = 0; i < (int)nodes.size()*2; i++) ydot[i + 2*nodes.size()] = y[i];
    for (int i = 0; i < (int)nodes.size()*2; i++) ydot[i] = acc[i] / node_masses[div(i,2).quot];
}

vector<double> integrate(double phases[])
{
    /* chain together multiple 1 second long integrations and measure position every 2 seconds, then take the
       average displacement and rotation of the measurements */
    double t = 0; vector<double> res, init, previousPos; int istate = 1; LSODA lsoda;
    init = y_init; previousPos = vector<double>(init.begin() + ncnt/2, init.end());
    double t_max = 10., step_size = 1.; vector<double> comx, comy, ang; double angSum = 0;
    for (double i = 1.; i <= t_max; i += step_size) {
        lsoda.lsoda_update(yprime, y_init.size(), init, res, &t, i * step_size, &istate, phases, 1e-4, 1e-4);
        if ((int)i % 2 == 0) {
            vector<double> currentPos = vector<double>(res.begin() + ncnt/2 + 1, res.end());
            double angDiff = angLst(previousPos,currentPos);
            ang.push_back(angDiff); angSum += angDiff;
            array<double,2> c1 = com(previousPos), c2 = com(currentPos); // can be optimised (redundant)
            array<double,2> corrected = rotate(c2[0] - c1[0], c2[1] - c1[1], -angSum);
            comx.push_back(corrected[0]); comy.push_back(corrected[1]);
            previousPos = vector<double>(currentPos.begin(), currentPos.end());
        }
        init = vector<double>(res); res = vector<double>();
    }
    vector<double> stats;
    for (int i = 0; i < blockcount; i++) stats.push_back(phases[i]);
    stats.push_back(accumulate(comx.begin()+1,comx.end(),0.)/(double)(comx.size()-1));
    stats.push_back(accumulate(comy.begin()+1,comy.end(),0.)/(double)(comy.size()-1));
    stats.push_back(accumulate(ang.begin()+1,ang.end(),0.)/(double)(ang.size()-1));
    return stats;
}

/**
 * create a 2d phasemap by iterating the phases at idx1 and idx2 of the phases[] array with step size delta and leaving the others
 * constant (just insert 0 or any value at idx1 and idx2 of phase array) and writes it to 'filename'
 */
void simple_phasemap(double delta, int idx1, int idx2, double phases[], string filename)
{
    vector<vector<double>> res;
    for (double x = 0.; x <= 2.; x += delta) {
        for (double y = 0.; y <= 2.; y += delta) {
            phases[idx1] = x; phases[idx2] = y;
            res.push_back(integrate(phases));
        }
        cout << "x=" << x << endl;
    }
    cout << "write phasemap to file: " << filename << endl;
    ofstream myfile;
    myfile.open(filename, ofstream::trunc);
    for (vector<double> el : res) {
        for (double d : el) {
            myfile << d << ",";
        }
        myfile << endl;
    }
    myfile.close();
    cout << "wrote phasemap file with " << res.size() << " entries." << endl;
}

int main(int argc, const char* argv[])
{
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    double phi[] = {0.0,0.0,0.0};
    simple_phasemap(2./40.,1,2,phi,"11_10_phasemap.txt");  // file standardly written to 'outfiles' directory because executable is stored there
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "Time taken " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " seconds" << endl;
    return 0;
}
