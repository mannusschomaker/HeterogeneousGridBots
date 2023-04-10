import os, time
import numpy as np 
import pandas as pd 
import itertools, functools 
import multiprocessing as mp 
from numba import njit, prange, jit
from datetime import datetime
from mass_spring_model import MassSpringModel, ModelParameters
from integrate import integrate_model, faster_integrate2, make_yprime2
from simulation_util import prepare_em_standard 


def rot(v,theta):
    c, s = np.cos(theta), np.sin(theta); R = np.array(((c, -s), (s, c))); return np.dot(R,v)


def displacement_measurement(phi, model, dummy_mat, integration_func=integrate_model, **kwargs):
	model.set_phases(phi)
	_, sol, _ = integration_func(model, em_initial=dummy_mat, learning=False, **kwargs)
	U = np.append([sol.t], sol.y[2*len(model.nodes):,:],axis=0).T
	sol_col = np.array([[f"x{i}", f"y{i}"] for i in range(len(model.nodes))]).flatten().tolist()
	df = pd.DataFrame(U,columns=["t"] + sol_col)
	df = df.assign(comx=df[sol_col[::2]].mean(axis=1),comy=df[sol_col[1::2]].mean(axis=1))
	df = df.assign(vx=[0] + [(df.comx[i]-df.comx[i-1])/(df.t[i]-df.t[i-1]) for i in range(1,len(df.comx))])
	shifted = np.array([x.to_numpy()-df[["comx","comy"]].to_numpy() for x in np.split(df[sol_col],len(model.nodes),axis=1)])
	df = df.assign(angle=(-np.mean([np.unwrap([np.arctan2(*r)-np.arctan2(*v[0]) for r in v]) 
	                                for v in shifted],axis=0))) # no mod 2 pi here because small change, also everything in radiants 
	angl = df["angle"].tolist()[80::20]
	corrected = np.transpose([rot(x,-(angl[i]-df["angle"].loc[0])) for i,x in enumerate(np.diff(df[["comx","comy"]].to_numpy()[80::20,:],axis=0))])
	displacement = corrected.mean(axis=1).tolist()
	#print("y",phi,displacement)
	return list(phi) + [np.diff(df["angle"][80::20]).mean(),displacement[0],displacement[1]]


@jit(parallel=True)  # TODO: nopython + this is not actually parallel yet
def _numba_map(func, phase_list):
	result = np.zeros((len(phase_list),len(phase_list[0])+3))
	for i in prange(len(phase_list)):
		result[i,:] = func(phase_list[i])
	return result


def displacement_map(shape, amplitudes, n, idx_fixed, name, cube_boundaries=[0,ModelParameters.t_cycle],faster=False):  # note that idx_fixed = False results in varying all indices
	# precompiled_data = yprime.address, y0 if using faster_integrate2
	# NOTE: cube boundaries != [0,t_cycle] is only used in generating the hypercube vertices (n=2 i.e. points at 0, boundary) for interpolation simulating periodic boundary conditions
	t1 = time.perf_counter()
	model = MassSpringModel(f"dummy", shape, 10 * ModelParameters.t_cycle) 
	dummy_mat = prepare_em_standard(len(model.blocks))
	dummy_mat[:,12] = amplitudes; dummy_mat[:,10] = np.full(len(model.blocks),0)
	space = [np.linspace(cube_boundaries[0],cube_boundaries[1],n) for _ in range(len(model.blocks))]
	if idx_fixed != False: space[idx_fixed] = [0]
	pool = mp.Pool()
	if not faster:
		res = list(pool.map(functools.partial(displacement_measurement,model=model,dummy_mat=dummy_mat,
			integration_func=integrate_model),itertools.product(*space)))
	else:
		yprime, y0 = make_yprime2(model)  # TODO: cfunc parallelisation problems
		res = _numba_map(functools.partial(displacement_measurement,model=model,dummy_mat=dummy_mat,
			integration_func=faster_integrate2, precompiled_yprime_addr=yprime.address, 
			y0=y0,tsteps=np.linspace(0,model.t_end,model.t_end*10)),np.array(list(itertools.product(*space))))
		
	df = pd.DataFrame(res,columns=[f"phase {i}" for i in range(len(model.blocks))] + ["d_theta","d_comx","d_comy"])
	now = datetime.now()
	dt_string = now.strftime("%d_%m_%Y")
	path = os.path.join(os.path.dirname(__file__), "..", "data", "experiment_files", dt_string)
	if not os.path.exists(path):
		os.makedirs(path)
	df.to_csv(os.path.join(path,f"displacement_map_{name}.csv"))
	print(f"Completed after {time.perf_counter() - t1:0.4f} seconds.")

