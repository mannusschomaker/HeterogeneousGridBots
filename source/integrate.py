import time 
import numpy as np  
from numba import njit, cfunc, carray
from numba.typed import List as TypedList
from numbalsoda import lsoda_sig, lsoda, dop853
from numpy.linalg import norm 
from scipy.integrate import solve_ivp
from mass_spring_model import ModelParameters


# physics functions
@njit(fastmath=True)
def actuation_length(t_current, ac): 
    # t_current is the local unit time now; ac = [A, f, t_cycle, t_motion, t_fixed, d_1]
    t = t_current % ac[2] - ac[3]/4; delta_l = 0.0035  # in m, you get this expansion on each side
    omega = ac[1]*2*np.pi
    if t < ac[3]/4:  # extension rise
        return ac[0] * (np.sin(omega * t) + 1) * delta_l
    elif t >= ac[3]/4 and t < (ac[3]/4 + ac[5]):  # extension plateau 
        return ac[0] * (np.sin(omega * ac[3]/4) + 1) * delta_l
    elif t >= (ac[3]/4 + ac[5]) and t < (3/4 * ac[3] + ac[5]):  # extension fall
        return ac[0] * (np.sin(omega * (t - ac[5])) + 1) * delta_l
    elif t >= (3/4 * ac[3] + ac[5]) and t < (3/4 * ac[3] + ac[4]):  # extension low plateau (contracted)
        return ac[0] * (np.sin(omega * ac[3] * 3/4) + 1) * delta_l
    elif t > (ac[3] * 3/4 + ac[4]):  # extention rise back to zero
        return ac[0] * (np.sin(omega * (t - ac[4])) + 1) * delta_l


@njit(fastmath=True)
def block_centres(U, blocks):
    c = np.zeros((len(blocks),2))
    for b in range(len(blocks)): 
        c[b] = np.array([np.mean(np.array([U[2*v] for v in blocks[b]])),   
                         np.mean(np.array([U[2*v+1] for v in blocks[b]]))])
    return c


@njit(fastmath=True)
def calculate_Fk_vector(l0, nodes, elements, U):  # spring force for each element
    Fk = np.zeros((len(elements),2))
    Fn = np.zeros(2*len(nodes))  # force exerted by the springs on the nodes 
    for i, el in enumerate(elements): 
        diff = np.array([U[2*el["to"]], U[2*el["to"] + 1]]) - \
            np.array([U[2*el["from"]], U[2*el["from"] + 1]])
        Fk[i] = (norm(diff) - l0[i]) * (el["k"]) * (diff/norm(diff))
        Fn[2*el["from"]] += Fk[i][0]
        Fn[2*el["from"] + 1] += Fk[i][1]
        Fn[2*el["to"]] -= Fk[i][0]
        Fn[2*el["to"] + 1] -= Fk[i][1]
    return Fn


@njit(fastmath=True)
def calculate_Ff_vector(V, V_constants, F_constants):  # slip-stick friction  
    v_length = np.where(V[::2]+V[1::2]==0,1,np.sqrt(V[::2]**2+V[1::2]**2))
    Ff = np.zeros(len(V))
    Ff[::2] += -(2.3316439*(F_constants[2])*np.exp(-(v_length/V_constants[2])**2)*(v_length/V_constants[2]) + F_constants[1]*np.tanh(v_length/V_constants[1]) + v_length* V_constants[3] ) * (V[::2]/v_length)
    Ff[1::2] += -(2.3316439*(F_constants[2])*np.exp(-(v_length/V_constants[2])**2)*(v_length/V_constants[2]) + F_constants[1]*np.tanh(v_length/V_constants[1]) + v_length* V_constants[3] ) * (V[1::2]/v_length)
    return Ff


# learning functions
@njit(fastmath=True)
def calculate_score_function(pos, light_source_pos_x):  # light intensity given position
    return (1/((np.sqrt((light_source_pos_x - pos[0])**2 + 1**2)) ** 2)) # 1/r^2 including vertical distance + it's a wall not point


@njit(fastmath=True)
def flaky_light(current_phase, best_phase, current_score, best_score):
    if current_score > best_score:
        return current_score, current_phase
    else:
        return current_score, best_phase


@njit(fastmath=True)
def update_event_matrix(event_matrix, t, event_data, pos, actuation_constants, event_constants, times_switch_light):
    # event_matrix = [unit][ 0 = local time of unit, 1 = current_phase, 2 = memory_phase, 3 = memory_score, 4 = current_event, 
    # 5 = time_of_next_event, 6 = distance_ref (not in use), 7 = light_ref, 8 = current_score, 9 = ds (learning step), 
    # 10 = n_transient_cycles, 11 = num complete learning cycles, 12 = amplitude "constant" A]
    # event_constants = [0 = n_transient_cycles, 1 = phase_transition, 2 = ds]
    # actuation_constants = [0 = A, 1 = f, 2 = t_cycle , 3 = t_motion, 4 = t_fixed, 5 = d_1]
    
    flag = False 
    if len(times_switch_light) > 0:
        if times_switch_light[0] <= t:
            times_switch_light.pop(0); event_matrix[:,13] = -event_matrix[:,13]; flag = True  # flip light source
    for index in range(len(event_matrix)):
        if event_matrix[index][4] == 0:  # normal actuation 
            event_matrix[index][0] =  t - event_matrix[index][2]
            if event_matrix[index][5] < t:
                event_matrix[index][4] = 1
                event_matrix[index][5] = event_matrix[index][5] + (actuation_constants[2] * event_matrix[index][10]) + actuation_constants[2] - event_constants[1]
        else: 
            event_matrix[index][0] =  t - event_matrix[index][1]

        if event_matrix[index][4] == 1:  # first measurement 
            if event_matrix[index][5] <= t:
                event_matrix[index][7] = calculate_score_function(pos[index], event_matrix[0][13])
                event_matrix[index][4] = 2
                event_matrix[index][5] = event_matrix[index][5] + actuation_constants[2]  # postpone next measurement
                flag = True 
                
        elif event_matrix[index][4] == 2:  # second measurement 
            if event_matrix[index][5] <= t:
                event_matrix[index][8] =  calculate_score_function(pos[index], event_matrix[0][13]) - event_matrix[index][7]  # new - reference score
                # learn and overwrite phases  
                event_matrix[index][3], event_matrix[index][2] = flaky_light(event_matrix[index][1], event_matrix[index][2], event_matrix[index][8], event_matrix[index][3])
                event_matrix[index][1] = event_matrix[index][2] + (np.random.rand() - 0.5) * actuation_constants[2] * event_matrix[index][9]   # ds is how much phase can be perturbed
                # update the next event to be the phases transition
                event_matrix[index][4] = 0
                event_matrix[index][11] += 1 # number of completed learning cycles increases 
                event_matrix[index][5] += event_constants[1] + (event_matrix[index][1] - event_matrix[index][2])  # phase transition
                flag = True 

        elif event_matrix[index][4] == 3:  # only for the beginning of the simulation
            event_matrix[index][0] = 0
            if event_matrix[index][1] <= t:
                event_matrix[index][4] = 1
                event_matrix[index][5] = event_matrix[index][1] + actuation_constants[2]
        
    if flag:  # ugly, but numba requires
        ed_row_new = TypedList(); ed_row_new.append(t)
        for x in event_matrix:  # NOTE: if actuation cycle were to change over time, this couuld become messy
            for y in [x[1] % actuation_constants[2], x[2] % actuation_constants[2], x[8], x[3], x[11], x[9], x[10], x[12]]:
                ed_row_new.append(y)
        ed_row_new.append(event_matrix[0][13])  # x position of light source
        event_data.append(ed_row_new)
    return event_matrix


# overall
@njit(fastmath=True)
def Yprime(t, Y, Y0, nodes, elements, masses, blocks, event_matrix, event_data, learning, V_constants, F_constants, actuation_constants, event_constants, times_switch_light):
    U = Y[int(len(Y)/2):len(Y)]
    V = Y[0:int(len(Y)/2)]
    if learning:  # numba requires to update external parameter by reference
        event_matrix = update_event_matrix(event_matrix, t, event_data,
            block_centres(U, blocks), actuation_constants, event_constants, times_switch_light) 
    l0 = np.zeros(len(elements))  # new equilibrium lengths including actuation 
    act_const = np.copy(actuation_constants)  # actuation "constants" (now allowed to change per unit)
    for i in range(len(elements)): 
        if elements[i]["actuator"] == 1:
            act_const[0] = event_matrix[elements[i]["parent"]][12]  # amplitude A
            if learning:
                l0[i] = elements[i]["l0"] + actuation_length(event_matrix[elements[i]["parent"]][0], act_const) 
            else:
                l0[i] = elements[i]["l0"] + actuation_length(t - elements[i]["phase"], act_const)         
        else:               
            l0[i] = elements[i]["l0"] 

    F_total = (calculate_Fk_vector(l0, nodes, elements, U) + calculate_Ff_vector(V, V_constants, F_constants)) / masses
    res = np.zeros(2 * len(F_total)); res[:len(F_total)] = F_total; res[len(F_total):] = V
    return res 


@njit(fastmath=True)
def set_seed(s): 
    np.random.seed(s)


def integrate_model(model, em_initial, learning=True, summary_function=lambda x, y, z: (x,y,z), times_switch_light=[0.0],seed=0):
    if seed > 0:
        set_seed(seed); print(em_initial)
    strc = model.to_struct()
    edata = TypedList()
    edata.append(TypedList([0.0]))  # TODO: don't hardcode n*len(model.blocks) (number of differnt attributes per unit)
    edata.pop()
    t_start = time.perf_counter()
    times_switch_light2 = TypedList(times_switch_light)
    sol = solve_ivp(Yprime, [0, model.t_end], model.Y0, args=(strc[0], strc[1], strc[2], strc[3], strc[4],
        em_initial, edata, learning, ModelParameters.V_constants, ModelParameters.F_constants, 
        ModelParameters.actuation_constants, ModelParameters.event_constants, times_switch_light2), 
        t_eval=model.t_eval, max_step=0.3)
    print(f"Integrated in {time.perf_counter() - t_start:0.4f} seconds")
    return summary_function(model, sol, [[x for x in r] for r in edata])  # need to convert datatype for multiprocessing pickle


def make_yprime2(model):
    """
    Wrapper to produce a pre-compiled integration function (rhs of diff. eq.) for integrating the given mass-spring model. Use this function ONCE.
    """
    ysize = len(model.nodes)*4; nel = len(model.elements)
    y0 = np.array([[0,0] for _ in range(len(model.nodes))] + [v.position for v in model.nodes]).flatten() 
    ci = []; cj = []; cd = []; prnts = []
    for i, el in enumerate(model.elements):
        ci.append(model.nodes.index(el.node_from)); cj.append(model.nodes.index(el.node_to))
        if el.is_actuator: 
            cd.append(i); prnts.append(model.blocks.index(el.parent_block))
    ci = np.array(ci); cj = np.array(cj); cd = np.array(cd); prnts = np.array(prnts)
    blck = np.array([[model.nodes.index(v) for v in b.nodes] for b in model.blocks])
    masses = model.masses
    ac = ModelParameters.actuation_constants; ec = ModelParameters.event_constants; sc = np.array([el.spring_constant for el in model.elements])
    fc = ModelParameters.F_constants; vc = ModelParameters.V_constants; ell = ModelParameters.element_length
    l0_ = np.full(nel,ell); l0_[cd] = np.full(len(cd),np.linalg.norm([ell,ell]))
    
    @cfunc(lsoda_sig,nopython=True)  # TODO vectorisation 
    def yprime(t, y, dy, info):  # info = [phases for all the blocks]
        y_ = carray(y,(ysize,),dtype=np.float64); V = y_[:ysize//2]; U = y_[ysize//2:]; pos = U.reshape((ysize//4,2))  # TODO: can be optimised
        info_ = carray(info,(nel,),dtype=np.float64); l0 = l0_.copy()
        l0[cd] = np.array([l0[cd[i]] + actuation_length(t-info_[prnts[i]],ac) for i in range(len(prnts))]) # TODO: faster; doesn't respect alternative amplitude yet
        sep_vec = pos[ci,:] - pos[cj,:]; sep = np.array([np.linalg.norm(v) for v in sep_vec]); dL = sep - l0; 
        ax = (-sc * dL * sep_vec[:,0]) / sep; ay = (-sc * dL * sep_vec[:,1]) / sep
        acc = np.zeros((ysize//4,2));
        for i in range(len(ci)): acc[:,0][ci[i]] += ax[i]
        for i in range(len(ci)): acc[:,1][ci[i]] += ay[i]
        for i in range(len(ci)): acc[:,0][cj[i]] += -ax[i]
        for i in range(len(ci)): acc[:,1][cj[i]] += -ay[i]
        acc = acc.reshape(ysize//2,); acc += calculate_Ff_vector(V,vc,fc); acc /= masses
        for i in range(ysize//2): dy[i] = acc[i]
        for i in range(ysize//2,ysize): dy[i] = V[i-(ysize//2)]

    return yprime, y0 


class SolPretender: 
    def __init__(self,y,t):
        self.y = y
        self.t = t


def faster_integrate2(model, precompiled_yprime_addr=None, y0=None, tsteps=None, **kwargs): 
    phi = np.array([b.phase for b in model.blocks],dtype=np.float64)
    t1 = time.perf_counter()
    usol, success = lsoda(precompiled_yprime_addr, y0, tsteps, data=phi)
    print(f"Integrated in {time.perf_counter() - t1} seconds. Success: {success}")
    return model, SolPretender(usol.T,tsteps), []