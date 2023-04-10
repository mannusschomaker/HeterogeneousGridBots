import numpy as np
import random
from collections import defaultdict  


class ModelParameters:
	element_length = 0.046  # in m - measured quantity
	m = 0.075
	spring_constant = 500

	# actuation parameters
	A = 1  # scaling factor - if A=1, then spring extension range is 2 (sides) * 3.5mm (measured per side)
	f = 0.85
	alpha = 0.33
	t_cycle = 2
	t_motion = 1/f
	t_fixed = t_cycle - t_motion
	d_1 = (t_cycle * alpha) - t_motion/2
	actuation_constants = np.array([A,f,t_cycle,t_motion,t_fixed,d_1])

	# friction parameters
	Fcoulomb = 0.172
	Fstatic = 0
	Fbrk = 0.172 #Fcoulomb + Fstatic
	F_constants = np.array([Fbrk, Fcoulomb, Fstatic])
	Vbrk = 0.01
	Vcoul = Vbrk/10
	Vst = Vbrk * np.sqrt(2)
	B = 2  # 4
	V_constants = np.array([Vbrk, Vcoul, Vst, B])

	# event model parameters
	time_in_phase = 1.0
	n_transient_cycles = 1.0
	ds = 0.1
	save_event_data = 0.0
	phase_transition = (actuation_constants[3]*(3/4)) + actuation_constants[5] + ((t_fixed - d_1)/2.0)
	event_constants = np.array([n_transient_cycles, phase_transition, ds, save_event_data]).astype(float)


class Node:
	def __init__(self, position):
		self.position = position  # np.array([x,y])
		self.mass = 0

	def __eq__(self, other):  
		return np.array_equal(self.position, other.position)

	def __repr__(self):
		return str(self.position)

	def __hash__(self):
		return hash(str(self))


class Spring:
	def __init__(self, node_from, node_to, is_actuator=False, adjoints_blocks=False, length=ModelParameters.element_length, spring_constant=ModelParameters.spring_constant):
		self.length = length 
		self.spring_constant = spring_constant
		self.node_from = node_from
		self.node_to = node_to 
		self.is_actuator = is_actuator
		self.adjoints_blocks = adjoints_blocks
		self.parent_block = None  # set during initialisation form matrix

	def __eq__(self, other): 
		return (self.node_from == other.node_from) and (self.node_to == other.node_to)

	def __repr__(self):
		return "({},{})".format(str(self.node_from),str(self.node_to))


class BlockUnit: 
	def __init__(self, upper_left_pos, spring_length=ModelParameters.element_length, phase=0):
		v1 = Node(upper_left_pos)
		v2 = Node(upper_left_pos + np.array([0, -spring_length]))
		v3 = Node(upper_left_pos + np.array([spring_length, 0]))
		v4 = Node(upper_left_pos + np.array([spring_length, -spring_length]))
		edges = [Spring(v1,v2,length=spring_length), Spring(v1,v3,length=spring_length), 
				Spring(v3,v4,length=spring_length), Spring(v2,v4,length=spring_length),
				Spring(v1,v4,True,length=np.power(2,0.5)*spring_length), 
				Spring(v2,v3,True,length=np.power(2,0.5)*spring_length)]
		self.nodes = [v1,v2,v3,v4]
		self.elements = edges
		self.spring_length = spring_length
		self.phase = phase 
	
	def get_centre(self):
		return self.nodes[0].position + [self.spring_length/2, -self.spring_length/2]


class MassSpringModel:
	def __init__(self, name, shape_block_matrix, t_end, rotation_angle=0):
		self.blocks = []
		self.nodes = []
		self.elements = []
		self.t_end = t_end
		self.t_eval = np.linspace(0,t_end,t_end*10)
		self.name = name  
		self.init_from_block_matrix(shape_block_matrix)
		self.set_node_masses_and_spring_constants()
		self.center_nodes()
		self.apply_rotation(rotation_angle)
		self.set_Y0_and_masses()
		self.initial_rotation = rotation_angle
		self.initial_CoM = self.get_CoM()

	def set_node_masses_and_spring_constants(self):
		# needs to be run after nodes and elements are generated 
		deg = defaultdict(int)
		for e in self.elements: 
			if e.is_actuator:
				deg[e.node_to] += 1
				deg[e.node_from] += 1
		for v in self.nodes: 
			v.mass = deg[v] * ModelParameters.m 
		for e in self.elements:
			if e.adjoints_blocks:
				e.spring_constant *= 2

	def init_from_block_matrix(self, matrix):
		for y in range(matrix.shape[0]):
			for x in range(matrix.shape[1]): 
				if matrix[y][x] == 1:
					block = BlockUnit(np.array([x*ModelParameters.element_length, 
						-y*ModelParameters.element_length]))
					self.blocks.append(block)
					for i in range(len(block.nodes)):
						if block.nodes[i] in self.nodes:
							block.nodes[i] = self.nodes[self.nodes.index(block.nodes[i])]
							for e in block.elements:
								if e.node_to == block.nodes[i]:
									e.node_to = self.nodes[self.nodes.index(block.nodes[i])]
								if e.node_from == block.nodes[i]:
									e.node_from = self.nodes[self.nodes.index(block.nodes[i])]
						else: 
							self.nodes.append(block.nodes[i])
					for elem in block.elements: 
						if elem in self.elements: 
							self.elements[self.elements.index(elem)].adjoints_blocks = True 
						else: 
							self.elements.append(elem)
					for elem in block.elements: 
						if elem.is_actuator:
							elem.parent_block = block  	
	
	def set_phases(self, phases): 
		for i in range(len(self.blocks)):
			self.blocks[i].phase = phases[i]

	def set_Y0_and_masses(self):
		self.Y0 = np.zeros(len(self.nodes) * 4)
		for i, v in enumerate(self.nodes):
			self.Y0[2*len(self.nodes) + 2*i] = v.position[0]
			self.Y0[2*len(self.nodes) + 2*i+1] = v.position[1]
		self.masses = np.zeros(2 * len(self.nodes))
		for i in range(len(self.nodes)):
			self.masses[2*i] = self.nodes[i].mass 
			self.masses[2*i + 1] = self.nodes[i].mass

	def calculate_bounding_box(self, margin=0.1):
		xvals = [v.position[0] for v in self.nodes]
		yvals = [v.position[1] for v in self.nodes]
		return ((min(xvals)-margin, max(xvals)+margin), (min(yvals)-margin, max(yvals)+margin))

	def center_nodes(self):  # TODO: make CoM based
		bb = self.calculate_bounding_box()
		offset = np.array([(bb[0][0] + bb[0][1])/2, (bb[1][0] + bb[1][1])/2])
		for i in range(len(self.nodes)): 
			self.nodes[i].position -= offset

	def apply_rotation(self, angle):  # angle in radiants 
		c,s = np.cos(angle), np.sin(angle)
		rotM = np.array(((c, -s), (s, c)))
		for v in self.nodes:
			v.position = np.dot(rotM, v.position.T).T		

	def get_element_struct(self): 
		e_struct = np.dtype([("from", np.int), ("to", np.int), ("k", np.float),
			("l0", np.float), ("actuator", np.int), ("phase", np.float), ("parent", np.int)]) 
		e = np.zeros(len(self.elements),dtype=e_struct)
		for i in range(len(self.elements)): 
			e[i]["from"] = self.nodes.index(self.elements[i].node_from)
			e[i]["to"] = self.nodes.index(self.elements[i].node_to)
			e[i]["actuator"] = (1 if self.elements[i].is_actuator else 0)
			e[i]["l0"] = self.elements[i].length 
			if self.elements[i].is_actuator:
				e[i]["parent"] = self.blocks.index(self.elements[i].parent_block)
				e[i]["phase"] = self.elements[i].parent_block.phase
			else: 
				e[i]["parent"] = -1  
				e[i]["phase"] = np.nan  # indeterminate because doesn't actuate
			e[i]["k"] = self.elements[i].spring_constant
		return e 

	def to_struct(self):
		blocks = np.array([[self.nodes.index(v) for v in b.nodes] for b in self.blocks])
		return [self.Y0, np.array([v.position for v in self.nodes]), self.get_element_struct(), self.masses, blocks]

	def to_struct2(self):  # not currently in use due to numba bug
		e_struct = np.dtype([("from", np.int), ("to", np.int), ("k", np.float),
			("l0", np.float), ("actuator", np.int), ("phase", np.float)]) # nodes as indices; phase of parent block
		struct = np.dtype([("Y0", np.float, (len(self.Y0),)), 
						   ("nodes", np.float, (len(self.nodes),2)),
						   ("elements", e_struct, (len(self.elements),)),  
						   ("masses", np.float, (len(self.masses)))])
		instance = np.zeros((), dtype=struct) 
		instance["Y0"] = self.Y0
		instance["nodes"] = [v.position for v in self.nodes]
		for i in range(len(self.elements)): 
			instance["elements"][i]["from"] = self.nodes.index(self.elements[i].node_from)
			instance["elements"][i]["to"] = self.nodes.index(self.elements[i].node_to)
			instance["elements"][i]["actuator"] = (1 if self.elements[i].is_actuator else 0)
			instance["elements"][i]["l0"] = self.elements[i].length 
			
			if self.elements[i].is_actuator:
				instance["elements"][i]["phase"] = self.elements[i].parent_block.phase
			else: 
				instance["elements"][i]["phase"] = np.nan  # indeterminate because doesn't actuate
			instance["elements"][i]["k"] = self.elements[i].spring_constant
		instance["masses"] = self.masses  
		return instance 

	def blocks_CoM_list(self):
		com_list = []
		for block in self.blocks: 
			com_list.append([np.mean([v.position[0] for v in block.nodes]), 
				np.mean([v.position[1] for v in block.nodes])])
		return np.array(com_list)

	def get_CoM(self):
		x = [v.position[0] for v in self.nodes]; y = [v.position[1] for v in self.nodes]
		return np.array([np.mean(x),np.mean(y)])

	def el2idx(self):  # used in configuration drawing
		lst = []
		for el in self.elements:
			lst.append((self.nodes.index(el.node_from), self.nodes.index(el.node_to)))
		return lst 


