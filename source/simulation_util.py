import time, pickle, csv, os, copy
import multiprocessing as mp 
import numpy as np 
import functools  
from datetime import datetime
from mass_spring_model import ModelParameters, MassSpringModel
from integrate import integrate_model
from render import AnimationWriter, PlottingTools


class OutputWriter: 
    @staticmethod
    def save_motion_data(model, sol, filename): 
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y")
        path = os.path.join(os.path.dirname(__file__), "..", "data", "experiment_files", dt_string)
        if not os.path.exists(path):
            os.makedirs(path)
        animation_data = [sol, model.nodes, model.elements]
        pickle.dump(animation_data, open(os.path.join(path, filename), "wb"))

    @staticmethod
    def save_event_data(edata_matrix, filename, num_units): 
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y")
        path = os.path.join(os.path.dirname(__file__),"..", "data", "experiment_files", dt_string)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, filename), "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["t"] + np.array([[f"phi {i}", f"mem. phi {i}", f"score {i}",
                f"mem. score {i}", f"learn cycle {i}", f"ds {i}", f"tcl {i}", f"A {i}"] 
                for i in range(num_units)]).flatten().tolist() + ["light_x"])
            for i in range(len(edata_matrix)): 
                writer.writerow(edata_matrix[i])

    @staticmethod
    def save_animation_file(model_name, frame_res=50, auto_axlim=False):
        an = AnimationWriter(AnimationWriter.get_path(datetime.now().strftime("%d_%m_%Y"), 
            model_name + ".p"), frame_resolution=frame_res, auto_axlim=auto_axlim)
        an.render_and_save(AnimationWriter.get_path(datetime.now().strftime("%d_%m_%Y"), 
            model_name + ".mp4", animation_file=True))


class SummaryFunctions: 
    @staticmethod   # TODO: deprecated, see jupyter notebook
    def com_from_sol(sol, skip_step):
        com = []; s = sol.y[:,::skip_step]
        for i in range(len(s[0])):
            U = s[len(s)//2:,i]; com.append(np.array([np.mean(U[::2]), np.mean(U[1::2])]))
        return np.array(com) 

    @staticmethod  # TODO: deprecated, see jupyter notebook
    def angle_from_sol(sol, com):
        avg_angle = []
        for i in range(len(sol[0])):
            U0 = sol[len(sol)//2:,0]; p0 = np.array(list(zip(U0[::2], U0[1::2]))) - com[0]
            U = sol[len(sol)//2:,i];  p = np.array(list(zip(U[::2], U[1::2]))) - com[i]
            angles = []
            for v1, v2 in zip(p0, p):
                angles.append(np.round(np.rad2deg(np.arccos(np.clip(np.inner(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)),-1,1))),2))
            if max(angles) - min(angles) >= 10: 
                print(f"warning: angle range is {max(angles) - min(angles):0.2f} degrees, values are {angles}, mean {np.mean(angles):0.2f}")
            avg_angle.append(np.mean(angles))
        return avg_angle  

    @staticmethod
    def save_wrapper(model, sol, edata, summary_function):  # use with functools.partial
        OutputWriter.save_motion_data(model, sol, model.name + ".p")
        OutputWriter.save_event_data(edata, model.name + "_event_data.csv")
        return summary_function(model, sol, edata)

    @staticmethod
    def edata_and_sparse_motion(model, sol, edata):  # currently in use for everything
        res = [] 
        #res.append(["t"] + np.array([[f"phi {i}", f"mem. phi {i}", f"score {i}",
        #        f"mem. score {i}", f"learn cycle {i}", f"ds {i}", f"tcl {i}", f"A {i}"] 
        #        for i in range(len(model.blocks))]).flatten().tolist() + np.array([[f"x{i}", f"y{i}"]
        #        for i in range(len(model.nodes))]).flatten().tolist())
        t_idx = 0 
        for row in edata:
            while t_idx < len(sol.t)-1 and abs(sol.t[t_idx] - row[0]) > 0.5:  # integration function (and thus event calculation) gets evaluated more often
                t_idx += 1 
            res.append(np.round(row + sol.y[len(sol.y)//2:,t_idx].tolist(),4))
        
        return np.array(res) 

    @staticmethod 
    def summary_x_displacement(model, sol, edata, num_measurements=10):  # precise only if num_measurements divides len(sol.y[0])
        com = SummaryFunctions.com_from_sol(sol, len(sol.y[0])//num_measurements)
        return com[:,0]

    @staticmethod  # TODO: deprecated, see jupyter notebook 
    def summary_avg_x_velocity(model, sol, edata, num_measurements=10):   # TODO: deprecated, see jupyter notebook
        d = SummaryFunctions.summary_x_displacement(model, sol, edata, num_measurements)
        v = [0]; step = len(sol.y[0])//num_measurements
        for i in range(1, len(d)):
            v.append((d[i] - d[i-1])/(sol.t[i * step] - sol.t[(i-1) * step]))
        return np.array(v) 

    @staticmethod  # TODO: deprecated, see jupyter notebook 
    def summary_angle(model, sol, edata, num_measurements=10):
        com = SummaryFunctions.com_from_sol(sol, len(sol.y[0])//num_measurements)
        shorter_sol = sol.y[:,::len(sol.y[0])//num_measurements]
        avg_angle = []
        for i in range(len(shorter_sol[0])):
            U0 = shorter_sol[len(shorter_sol)//2:,0]; p0 = np.array(list(zip(U0[::2], U0[1::2]))) - com[0]
            U = shorter_sol[len(shorter_sol)//2:,i];  p = np.array(list(zip(U[::2], U[1::2]))) - com[i]
            angles = []
            for v1, v2 in zip(p0, p):
                #angles.append(np.round(np.rad2deg(np.arccos(np.clip(np.inner(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)),-1,1))),2))
                theta = -np.arctan2(v1[1],v1[0]); c, s = np.cos(theta), np.sin(theta); R = np.array(((c, -s), (s, c)))
                v3 = np.dot(R, v2); angles.append(np.round(np.rad2deg(np.arctan2(v3[1],v3[0])),2))
            if max(angles) - min(angles) >= 10: 
                print(f"warning: angle range is {max(angles) - min(angles):0.2f} degrees, values are {angles}, mean {np.mean(angles):0.2f}")
            avg_angle.append(np.mean(angles))
        return np.array(avg_angle)  

    @staticmethod
    def summary_x_v_alpha(model, sol, edata, num_measurements=30):
        t = np.array(sol.t[::(len(sol.y[0])//num_measurements)])
        sx = SummaryFunctions.summary_x_displacement(model, sol, edata, num_measurements)
        sv = SummaryFunctions.summary_avg_x_velocity(model, sol, edata, num_measurements)
        salpha = SummaryFunctions.summary_angle(model, sol, edata, num_measurements)
        return np.array([t, sx, sv, salpha])


def prepare_em_standard(block_count):
    em_initial = np.zeros((block_count,14),dtype=float)
    init_phi = np.random.rand(block_count)*2  # randomly set initial phases 
    em_initial[:,1] = init_phi
    em_initial[:,2] = init_phi
    em_initial[:,4] = np.full(block_count, 3)  # event number for init
    em_initial[:,9] = np.full(block_count, ModelParameters.event_constants[2])  # ds 
    em_initial[:,10] = np.full(block_count, ModelParameters.event_constants[0])  # n_transient_cycles
    em_initial[:,11] = np.zeros(block_count)  # number completed learning cycles 
    em_initial[:,12] = np.full(block_count, ModelParameters.actuation_constants[0])  # A
    em_initial[:,13] = np.full(block_count, -20)  # x-coordinate of light source (doesn't depend on unit, stored here for convenience), starts negative since light list can't be empty
    return em_initial


def prepare_em_matrix_based(block_matrix, par_dict):
    # par_dict has the form {"parameter name": matrix of values for blocks}
    # parameter names: ds, tcl, A
    n_blocks = np.array(block_matrix).flatten()
    # TODO


def simulation_ensemble(model, event_matrix, num_simulations=10, summary_function=SummaryFunctions.summary_x_v_alpha,
        new_init_phases_each_time=True, times_switch_light=[0.0]):
    print(f"Starting {num_simulations} simulations...")
    t1 = time.perf_counter()
    parameters = []
    for i in range(num_simulations):
        mod = copy.copy(model); mod.name += "_" + str(i)
        mat = copy.copy(event_matrix)
        if new_init_phases_each_time:
            mat[:,1] = np.random.rand(len(model.blocks))*2
        parameters.append((mod, mat, True, summary_function, times_switch_light))

    pool = mp.Pool(mp.cpu_count())
    res = pool.starmap(integrate_model, parameters)
    print(f"Completed after {time.perf_counter() - t1:0.4f} seconds.")
    return res 


def sweep_baseline_transcycl_ds():  # TODO: metafile 
    t1 = time.perf_counter()
    n_measurements = 10; nsim = 3; n_standard_learning_cycles = 10#300
    path = os.path.join(os.path.dirname(__file__), "..", "data", "sweeps", datetime.now().strftime("%d_%m_%Y"), "baseline_tc_ds")
    if not os.path.exists(path):
        os.makedirs(path)
    for n_trans_cycl in [1.0, 2.0, 0.0]:
        for ds in np.arange(0.15,0.31,0.15):
            print(f"ds={ds}, n_trans_cycles={n_trans_cycl}. ", end="")
            model = MassSpringModel(f"baseline_{n_trans_cycl}tc_{ds}ds", np.array([[0,1],[1,1]]), 
                int(n_standard_learning_cycles * ModelParameters.t_cycle * (ModelParameters.n_transient_cycles + 2)), rotation_angle=np.pi/4) 
            mat = prepare_em_standard(len(model.blocks))
            mat[:,9] = np.full(len(model.blocks), ds)
            mat[:,10] = np.full(len(model.blocks), n_trans_cycl)
            solutions = simulation_ensemble(model, mat, num_simulations=nsim, 
                summary_function=functools.partial(SummaryFunctions.summary_x_v_alpha, num_measurements=n_measurements))
            f = open(os.path.join(path,f"{nsim}sim_{int(n_standard_learning_cycles * ModelParameters.t_cycle * (ModelParameters.n_transient_cycles + 2))}s" + 
                f"_{n_measurements}m_{n_trans_cycl}tc_{ds}ds.p"), "wb")
            pickle.dump(solutions, f)
    print(f"Sweep finished after {time.perf_counter() - t1:0.4f} seconds.")


def sweep_triangle_ds(n_transient_cycles=1, nsim=50, n_standard_learning_cycles=500, sweep_add_string=""):
    t1 = time.perf_counter()
    path = os.path.join(os.path.dirname(__file__), "..", "data", "sweeps", datetime.now().strftime("%d_%m_%Y"), f"triangle_ds{sweep_add_string}")
    if not os.path.exists(path):
        os.makedirs(path)
    ds_arr = [[0.00,0.01],[0.00,0.04],[0.00,0.16],  # [0.00 (random), 0.01, 0.04 (baseline), 0.16]
              [0.01,0.01],[0.01,0.04],[0.01,0.16],  # [0.00,0.00] not used because totally random 
                          [0.04,0.04],[0.04,0.16]   # left upper matrix due to triangle symmetry 
                                     ,[0.16,0.16]]
    t_steps = int(n_standard_learning_cycles * ModelParameters.t_cycle * (n_transient_cycles + 2))
    print(f"Sweeping {2 * len(ds_arr) * 3} parameter configurations, {nsim} simulations each, {t_steps}s simulated time each:")
    for angle in [np.pi * 5/4, np.pi * 1/4]:     
        for ds_pair in ds_arr:   
            geom_dist = [[ds_pair[0], ds_pair[1], 0.04], [ds_pair[0], 0.04, ds_pair[1]], [ds_pair[1], 0.04, ds_pair[0]]]
            for g_idx in range(len(geom_dist)):
                print(f"angle={np.rad2deg(angle)}, ds={geom_dist[g_idx]}.", end="")
                model = MassSpringModel(f"triangle_ds", np.array([[0,1],[1,1]]), t_steps, rotation_angle=angle) 
                #print([(model.nodes.index(el.node_from), model.nodes.index(el.node_to)) for el in model.elements])
                mat = prepare_em_standard(len(model.blocks))
                mat[:,9] = geom_dist[g_idx]
                mat[:,10] = np.full(3, n_transient_cycles)
                solutions = simulation_ensemble(model, mat, num_simulations=nsim, 
                    summary_function=SummaryFunctions.edata_and_sparse_motion)
                f = open(os.path.join(path,f"{nsim}sim_{t_steps}s_" +
                        f"{geom_dist[g_idx][0]:0.2f}_{geom_dist[g_idx][1]:0.2f}_{geom_dist[g_idx][2]:0.2f}ds_{np.rad2deg(angle):0.2f}ang.p"), "wb")
                pickle.dump(solutions, f)
                if g_idx == 0:
                    f = open(os.path.join(path,f"{nsim}sim_{t_steps}s_" +
                        f"{geom_dist[g_idx][1]:0.2f}_{geom_dist[g_idx][0]:0.2f}_{geom_dist[g_idx][2]:0.2f}ds_{np.rad2deg(angle):0.2f}ang.p"), "wb")
                    pickle.dump(solutions, f)
    print(f"Sweep finished after {time.perf_counter() - t1:0.4f} seconds.")


def sweep_triangle_amp(n_transient_cycles=1, nsim=30, ds=[0.04,0.04,0.04], n_standard_learning_cycles=600, angle=5/4 *np.pi, sweep_add_string=""):
    t1 = time.perf_counter()
    path = os.path.join(os.path.dirname(__file__), "..", "data", "sweeps", datetime.now().strftime("%d_%m_%Y"), f"triangle_amp{sweep_add_string}")
    if not os.path.exists(path):
        os.makedirs(path)
    t_steps = int(n_standard_learning_cycles * ModelParameters.t_cycle * (n_transient_cycles + 2))
    print(f"Sweeping 64 parameter configurations, {nsim} simulations each, {t_steps}s simulated time each:")
    for amp1 in [0,0.5,1,3]:
        for amp2 in [0,0.5,1,3]:
            for amp3 in [0,0.5,1,3]:
                if amp1 == amp2 and amp2 == amp3: continue
                print(f"amplitudes {amp1, amp2, amp3}")
                model = MassSpringModel(f"triangle_amp", np.array([[0,1],[1,1]]), t_steps, rotation_angle=angle) 
                mat = prepare_em_standard(len(model.blocks))
                mat[:,9] = ds
                mat[:,10] = np.full(3, n_transient_cycles)
                mat[:,12] = [amp1,amp2,amp3]  # (l,r,m) TODO: check which unit is which 
                solutions = simulation_ensemble(model, mat, num_simulations=nsim, 
                    summary_function=SummaryFunctions.edata_and_sparse_motion)
                f = open(os.path.join(path,f"{nsim}sim_{t_steps}s_" +
                    f"{amp1}_{amp2}_{amp3}am_{np.rad2deg(angle):0.2f}ang.p"), "wb")
                pickle.dump(solutions, f)     
    print(f"Sweep finished after {time.perf_counter() - t1:0.4f} seconds.")


def lightswitch_test(shape, ds, amp, angle, tcl, nsim=30, t_steps=12000, switch_times=[0.0,4000.0,8000.0]):
    t1 = time.perf_counter()
    path = os.path.join(os.path.dirname(__file__), "..", "data", "experiment_files", datetime.now().strftime("%d_%m_%Y"), f"switch_light")
    if not os.path.exists(path):
        os.makedirs(path)
    model = MassSpringModel(f"light_switch", np.array(shape), t_steps, rotation_angle=angle) 
    #print([[model.nodes.index(e.node_from), model.nodes.index(e.node_to)] for e in model.elements])
    mat = prepare_em_standard(len(model.blocks))
    mat[:,9] = ds; mat[:,10] = tcl; mat[:,12] = amp
    solutions = simulation_ensemble(model, mat, num_simulations=nsim, 
        summary_function=SummaryFunctions.edata_and_sparse_motion, times_switch_light=switch_times)
    f = open(os.path.join(path,f"{nsim}sim_{t_steps}s_" +
        f"{amp}am_{ds}ds_{switch_times}light.p"), "wb")
    pickle.dump(solutions, f)     

    print(f"Finished after {time.perf_counter() - t1:0.4f} seconds.")
