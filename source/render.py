import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import pickle
import numpy as np
import os


class AnimationWriter:
	@staticmethod
	def get_path(date, datafile_name, experiment_name="", shape="", learning_alg="", frame_resolution=1, mannus_convention=False, animation_file=False):
		if mannus_convention:
			path = os.path.join(os.path.dirname( __file__), "..", "mannus", "data",
				learning_alg, shape, experiment_name, date, datafile_name)
		else: 
			anim = "experiment_files"
			if animation_file: 
				anim = "animations"
			path = os.path.join(os.path.dirname(__file__), "..", "data", anim,
				date, datafile_name)
		return path 

	def __init__(self, input_filename, mannus_convention=False, auto_axlim=False, frame_resolution=1):
		self.frame_resolution = frame_resolution; self.mannus_convention = mannus_convention; self.auto_axlim = auto_axlim
		if self.mannus_convention:
			self.sol, self.nodes, self.elements, spring_list, global_spring, block_list = pickle.load(open(input_filename, "rb"))	
		else:
			self.sol, self.nodes, self.elements = pickle.load(open(input_filename, "rb"))	
		self.framecount = len(self.sol.y[0]) // self.frame_resolution
		self.__init_plot()
		
	def __init_plot(self, margin=0.1):
		self.fig = plt.figure()
		if self.auto_axlim:
			xlim = (np.amin(self.sol.y[::2]) - margin, np.amax(self.sol.y[::2]) + margin)
			ylim = (np.amin(self.sol.y[1::2]) - margin, np.amax(self.sol.y[1::2]) + margin)
			ax = plt.axes(xlim=xlim, ylim=ylim) 
		else:	
			ax = plt.axes(xlim=(-0.4, 1.2), ylim=(-0.8, 0.8)) 
		self.line, = ax.plot([], [], lw=2) 
		self.lines = [plt.plot([], [], lw=2, marker='o', c="g", zorder=3)[0] for _ in range(len(self.elements))]
		self.time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, zorder=2)
		plt.grid(True, linestyle='-.')

	def __init_animation(self):
		self.time_text.set_text('')
		self.line.set_data([], []) 
		return self.line, self.time_text

	def __animate(self, i):
		#if i % 100 == 0: print(i)
		self.time_text.set_text('time = %.1f' % self.sol.t[i * self.frame_resolution])
		U = self.sol.y[len(self.sol.y)//2:, i * self.frame_resolution]
		if self.mannus_convention:
			el_idx = [[self.elements[i][0], self.elements[i][1]] for i in range(len(self.elements))]
		else:
			el_idx = [[self.nodes.index(el.node_from), self.nodes.index(el.node_to)] for el in self.elements]
		for i, el in enumerate(el_idx): 
		    self.lines[i].set_data([U[2*el[0]], U[2*el[1]]], [U[2*el[0]+1], U[2*el[1]+1]])

		# TODO: mannus convention?
		#temp_nodes = self.sol.y[:, i * self.frame_resolution]  # current y data 
		#temp_nodes = temp_nodes[int(len(temp_nodes)/2):len(temp_nodes)]  # position displacement
		#for index, element in enumerate(self.elements):
		#	fromPoint, toPoint = self.__get_point_positions(temp_nodes, element)
		#	self.lines[index].set_data([fromPoint[0], toPoint[0]], [fromPoint[1], toPoint[1]])
		return self.lines, self.time_text

	def render_and_save(self, path):
		self.anim = animation.FuncAnimation(self.fig, self.__animate, init_func=self.__init_animation, frames=self.framecount)
		writer = animation.writers["ffmpeg"](fps=10, bitrate=140)
		if not os.path.exists(os.path.dirname(path)):
			os.makedirs(os.path.dirname(path))
		self.anim.save(path, writer=writer)
		

class PlottingTools:  # very rudimentary plotting functionality, for much more graphics, see jupyter notebooks in analysis folder  
	@staticmethod
	def draw_block_configuration(model): 
		fig = plt.figure()
		ax_lim = model.calculate_bounding_box()
		ax = plt.axes(xlim=ax_lim[0], ylim=ax_lim[1]) 
		lines = [plt.plot([], [], lw=2, marker='o', c="g", zorder=3)[0] for _ in range(len(model.elements))]
		for index, element in enumerate(model.elements):
			lines[index].set_data([element.node_from.position[0], element.node_to.position[0]], 
								  [element.node_from.position[1], element.node_to.position[1]])
			if element.is_actuator:
				lines[index].set_color("#FFD580")
				lines[index].set_alpha(0.4)
		for block in model.blocks: 
			plt.annotate(str(np.round(block.get_centre(),3)), xy=block.get_centre(), zorder=3, alpha=1, size=10)

		plt.show()

	@staticmethod
	def draw_positions_plot(model, sol, frame_step=1): 
		for i, line in enumerate(sol.y[int(len(sol.y)/2):len(sol.y),::frame_step]):
			if i%2 ==0: axis = "x"
			else: axis = "y"
			plt.plot(sol.t[::frame_step], line, label=axis+" "+str(i//2)) 
			plt.legend()
			plt.xlabel("Time")
			plt.ylabel("Position")
		plt.show()

	@staticmethod
	def draw_com_plot(sol):
		cx = []; cy = []
		for i in range(len(sol.y[0])):
			U = sol.y[int(len(sol.y)/2):len(sol.y), i]
			cx.append(np.mean(U[::2])); cy.append(np.mean(U[1::2]))
		plt.plot(sol.t, cx, label="CoM x")
		plt.plot(sol.t, cy, label="CoM y")
		plt.xlabel("Time")
		plt.ylabel("Displacement")
		plt.legend()
		plt.show()

	@staticmethod
	def draw_actuation_plot(model):
		for b in model.blocks:
			arr = np.linspace(0,2*ModelParameters.t_cycle, ModelParameters.t_cycle * 200)
			plt.plot(arr, np.array([actuation_length(x - b.phase, ModelParameters.actuation_constants) for x in arr]))
		plt.show()

	@staticmethod 
	def draw_phase_plot(event_data, block_count):
		ed = np.array(event_data)
		plt.plot(ed[:,0], ed[:,1:block_count+1])
		plt.show()


	