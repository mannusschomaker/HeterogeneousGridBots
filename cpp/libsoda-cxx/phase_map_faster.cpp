#include <iostream>
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
const double B = 2;
const double V_constants[] = {Vbrk, Vcoul, Vst, B};

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

const vector<vector<int>> shape = {{1}};//,0},{1,1}};
const vector<vector<double>> nodes = get_model_positions(shape);  // {{-0.05,0},{0.05,0}}; //
const vector<vector<int>> springs = get_model_springs(shape,nodes).first; // {{0,1}};
const vector<int> actuators = get_model_springs(shape,nodes).second; // {0.};
const vector<double> spring_constants = get_spring_constants(shape, springs, nodes);//{100.0};
const vector<double> node_masses = get_masses(springs, nodes); 
const vector<double> rest_lengths = get_rest_lengths(springs, nodes); 
vector<double> y_init = get_y0(nodes);


// integration-related functions
double actuation_length(double time) 
{
    //(void)ac; (void)omega; (void)delta_l; 
    //if (time < 5.0) return -0.01; 
    //return 0.01; 
    //return 0.01 * sin(time/5.0);// + sin(time)/1000.0; 
    
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
        exit(1);  
}

double friction_term(double x, double l) 
{
    return -(2.3316439 * F_constants[2] * exp(-pow(l/V_constants[2],2)) * (l / V_constants[2]) 
        +  F_constants[1] * tanh(l/V_constants[1])  +  l * V_constants[3]) * (x / l);
}

void yprime(double t, double* y, double* ydot, void* phi)
{
    (void)phi; 
    vector<double> acc(nodes.size() * 2, 0.0);  // force (acceleration in last loop) imparted on each node (component-wise)   
    vector<double> l0(rest_lengths);
    for (int i = 0; i < (int)actuators.size(); i++) {  // actuate (new l0 = rest len + time-dependent)
        l0[actuators[i]] += actuation_length(t - ((double*)phi)[div(i,2).quot]);
    }
    for (int i = 0; i < (int)l0.size(); i++) 
        cout << l0[i] << ",";
    cout << endl;
    int ix1, iy1, ix2, iy2;  // flattened-out indices of the components of start/end node of a spring
    double l, ax, ay;
    for (int i = 0; i < (int)springs.size(); i++) {  // Hooke's law
        ix1 = 2*nodes.size() + 2*springs[i][0]; iy1 = ix1+1; 
        ix2 = 2*nodes.size() + 2*springs[i][1]; iy2 = ix2+1;
        l = sqrt(pow(y[ix1]-y[ix2],2.0) + pow(y[iy1]-y[iy2],2.0));
        //cout << " l="<< l << ","; 
        ax = -spring_constants[i] * (y[ix1]-y[ix2]) * (l-l0[i]) / l;
        ay = -spring_constants[i] * (y[iy1]-y[iy2]) * (l-l0[i]) / l;
        //cout << ax << " - " << ay << endl;
        //cout << springs[i][0] << " -> " << springs[i][1] << endl; 
        acc[2*springs[i][0]] += ax; acc[2*springs[i][0] + 1] += ay; 
        acc[2*springs[i][1]] -= ax; acc[2*springs[i][1] + 1] -= ay; 
        // TODO: springs only once 
    }
    //cout << endl;
    
    for (int i = 0; i < (int)nodes.size(); i++) {  // friction
        ix1 = 2*i; iy1 = ix1 + 1; 
        l = sqrt(pow(y[ix1],2.0) + pow(y[iy1],2.0));  if (l==0) l = 1;
        //cout << " l="<< l << ","; 
        acc[2*i] += friction_term(y[ix1], l);
        acc[2*i+1] += friction_term(y[iy1], l);
    }
    // y0 - vel, pos    y - [vel, pos]   ydot - [acc, vel]    never read ydot
    for (int i = 0; i < (int)nodes.size()*2; i++) ydot[i + 2*nodes.size()] = y[i];  
    for (int i = 0; i < (int)nodes.size()*2; i++) ydot[i] = acc[i] / node_masses[div(i,2).quot];   
    
    cout << t << ",";
    //for (int i = 0; i < 4*(int)nodes.size(); i++) cout << ydot[i] << ",";
    //for (int i = 2*(int)nodes.size(); i < 4*(int)nodes.size(); i++) cout << y[i] << ",";
    //cout << endl << endl;
    for (int i = 0; i < 4*(int)nodes.size();i++) cout << y[i] << ","; 
    //for (double x : l0) cout << x << ",";
    cout << endl;
}

int integrate(double phases[])
{
    double t = 0; vector<double> res, init; int istate = 1; LSODA lsoda;
    //for (int i=0; i < 6; i++) cout << " | " << ac[i];
    //exit(0);
    //cout << endl << "init: "; for (double d : y_init) cout << d << " ";  
    //cout << endl;
    init = y_init; 
    /*for (double i = 1.; i < 10.; i += 1.) {
        lsoda.lsoda_update(yprime, y_init.size(), init, res, &t, i * t_cycle / 2., &istate, phases, 1e-5, 1e-5);
        init = vector<double>(res); res = vector<double>();
    }*/
    lsoda.lsoda_update(yprime, init.size(), init, res, &t, 1.0, &istate, phases, 1e-5, 1e-5);
    
    

    /*for (int iteration = 1; iteration < 20; iteration++) {
        lsoda.lsoda_update(yprime, y_init.size(), y_init, res, &t, (double)iteration * dt, &istate, phases);
        if (istate <= 0) {
            cerr << "error istate = " << istate << endl; exit(0);
        }
        cout << "t=" << t << endl;
        cout << "res: ";
        for (int i = 1; i < (int)res.size(); i++) cout << res[i] << " ";
        cout << endl;
    }*/
    //cout << endl << "tend=" << t << endl;
    //cout << "res: ";
    //for (int i = 1; i < (int)res.size(); i++) cout << res[i] << " ";
    //cout << endl;
    return 0;
}

template <typename RAIter>
int parallel_sum(RAIter beg, RAIter end)
{
    auto len = end - beg;
    if(len < 1000) {  
        return std::accumulate(beg, end, 0);
    }
    RAIter mid = beg + len/2;
    auto handle = std::async(std::launch::async, parallel_sum<RAIter>, mid, end);
    int sum = parallel_sum(beg, mid);
    return sum + handle.get();
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

    //chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    double phases[] = {0.0};//,0.3,0.6};
    assert((sizeof(phases)/sizeof(*phases) == actuators.size() / 2));
    integrate(phases);
    //chrono::steady_clock::time_point end = chrono::steady_clock::now();
    //cout << "|| Time taken (us)= "
    //     << chrono::duration_cast<chrono::microseconds>(end - begin).count() << endl;

    //cout << endl << "y0: ";
    //for (double x : y_init) cout << x << " ";
    //cout << endl; 

    //std::vector<int> v(10000, 1);
    //std::cout << "The sum is " << parallel_sum(v.begin(), v.end()) << '\n';

    return 0;
}
