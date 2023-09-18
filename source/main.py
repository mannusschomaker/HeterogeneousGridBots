import os 
import numpy as np
import pickle
import itertools
import time 
from datetime import datetime
from mass_spring_model import MassSpringModel, ModelParameters
from integrate import integrate_model, faster_integrate2
from simulation_util import prepare_em_standard, OutputWriter, SummaryFunctions
from render import PlottingTools, AnimationWriter
from simulation_util import simulation_ensemble


#nsim = 50; t_steps = 3000; path = "../data/sweeps/27_10_2022/triangle_ds/"
#for angle in [np.pi/4, np.pi*5/4]:
#    model = MassSpringModel(f"hom_ds_{np.round(np.rad2deg(angle))}", np.array([[0,1],[1,1]]), t_steps, rotation_angle=angle) 
#    mat = prepare_em_standard(len(model.blocks))
#    mat[:,9] = [0.04,0.04,0.04]; mat[:,10] = np.full(3, 1)
#    solutions = simulation_ensemble(model, mat, num_simulations=nsim, summary_function=SummaryFunctions.edata_and_sparse_motion)
#    if not os.path.exists(path): os.makedirs(path)
#    f = open(os.path.join(path,f"{nsim}sim_{t_steps}s_0.04_0.04_0.04ds_{np.rad2deg(angle):0.2f}ang.p"), "wb")
#    pickle.dump(solutions, f)

#model = MassSpringModel(f"light_test1", np.array([[0,1],[1,1]]), 1000, rotation_angle=np.pi/2) 
#mat = prepare_em_standard(len(model.blocks))
#mat[:,9] = [0.04,0.04,0.04]; mat[:,10] = np.full(3, 1); mat[:,12] = [1,1,1]
#m2, sol, edata = integrate_model(model,mat,times_switch_light=[0.0,500.0])
#OutputWriter.save_motion_data(model, sol, model.name + ".p")
#OutputWriter.save_event_data(edata, model.name + ".csv", len(model.blocks))
#OutputWriter.save_animation_file(model.name,auto_axlim=True)


from displacement_map import displacement_map 

#for amp in itertools.product(*[[0,0.25,0.5,1,2,4,8,16] for _ in range(3)]):
	#for amp in [[1,1,1],[1,1,3],[1,1,5],[1,1,0.5],[1,0.5,1],[1,3,1],[1,5,1],[1,0.5,0.5],[1,3,3],[1,5,5]]:
	#print("Testing amplitudes:",list(amp))
	#displacement_map(np.array([[0,1],[1,1]]),list(amp),50,2,f"triangle_{list(amp)}")
#displacement_map(np.array([[0,1],[1,1]]),[1,1,1],50,2,"faster1",faster=True)
	
def mat_str(arr): 
	s = ""
	for a in arr: 
		for b in a: 
			s += str(b)
		s += "_"
	return s[:-1] 


def main1(): 
	n_transient_cycles = 1; ds = 0.04; nsim = 30; n_standard_learning_cycles = 100
	shape = np.array([[1,1,1,1,1],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,0,1]])

	t1 = time.perf_counter()
	path = os.path.join(os.path.dirname(__file__), "..", "data", "sweeps", datetime.now().strftime("%d_%m_%Y"), f"{mat_str(shape)}_rotation_instability_amplitude")
	if not os.path.exists(path):
		os.makedirs(path)
	t_steps = int(n_standard_learning_cycles * ModelParameters.t_cycle * (n_transient_cycles + 2))
	print(f"Sweeping 8 parameter configurations for {mat_str(shape)}, {nsim} simulations each, {t_steps}s simulated time each:")
	#for j in range(10):
		#amp = np.random.normal(2.0,1.0,9).tolist()  # mean=2, var=1
	for amp in [[5,4,3,2,1,2,3,4,5],[1,2,3,4,5,4,3,2,1],[3,3,3,3,3,3,3,3,3]]:
		print(f"amplitude {amp}")
		model = MassSpringModel(f"{shape}_amp{amp}", shape, t_steps) 
		mat = prepare_em_standard(len(model.blocks))
		mat[:,9] = np.full(len(model.blocks),ds)
		mat[:,10] = np.full(len(model.blocks), n_transient_cycles)
		mat[:,12] = amp#np.full(len(model.blocks),amp)
		solutions = simulation_ensemble(model, mat, num_simulations=nsim, summary_function=SummaryFunctions.edata_and_sparse_motion)
		f = open(os.path.join(path,f"{mat_str(shape)}_{amp}amp_{nsim}sim_{t_steps}steps.p"), "wb")
		pickle.dump(solutions, f)     
	print(f"Sweep finished after {time.perf_counter() - t1:0.4f} seconds.")

if __name__ == '__main__':
	for a in [1,0.5,2]:
		for b in [1,0.5,2]: 
			for c in [1,0.5,2]: 
				for d in [1,0.5,2]: 
					print("map: ",a,b,c,d)
					M = np.array([[1,1,0],[0,1,1],[0,1,0]])
					displacement_map(M,[a,b,1,c,d],20,2,mat_str(M) + f"_map_{a}_{b}_1_{c}_{d}", faster=True)



	
