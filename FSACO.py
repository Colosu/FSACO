#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:16:05 2020

@author: colosu
"""

import xml.etree.ElementTree as ET
from threading import Thread
import time
import operator
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np

class splap:
	def __init__(self, file, features, node_to_children):
		self.file = file
		self.root = ET.parse(self.file).getroot()
		self.warnings = []
		self.prev = []
		self.excludes_list = []
		self.node_to_children = node_to_children
		probs = dict()
		for f in features.keys():
			probs[f] = self.getProb(f)
		self.features = features
		self.probs = probs
		
	def debug(self, text):
		"""
		Debug function. The text is shown into the std output stream.
		@param text : Not empty
		@rtype text : String for debugging
		"""
		print("{0}".format(text))

	def add_warning(self, tt, old, new):
		self.warnings
		self.warnings.append((tt,list(old),list(new)))
	
	def reduce (self, l):
		m = set(map(frozenset,l))
		l = list(map(list,m))

	def compute_denotational(self, root, featureP):
		"""
		This function defines the denotational semantics for a fodaAT term.
		@param root: not null
		@rtype root: xml element
		"""
		self.prev
		#print "computing {0}".format(root.tag)

		result = []
		if(root.tag=="xml"):
			#initial tag
			for i in list(root):
				result.append(self.compute_denotational(i, featureP))
		elif(root.tag=="checkmark"):
			#checkmark rule
			result = [[]]
		elif(root.tag=="mandatory_feature"):
			feature = root.get("name")

			for i in list(root):
				computation = self.compute_denotational(i, featureP)
				result.extend(computation)

			for i in result:
				if feature == featureP:
					i.append(feature)        #Agregamos el feature a cada elemento de la lista de result...
				else:
					if "*" not in i:
						i.append("*")

		elif(root.tag=="optional_feature"):
			feature = root.get("name")
			
			for i in list(root):
				computation = self.compute_denotational(i, featureP)
				result.extend(computation)
				
			for i in result:
				if feature == featureP:
					i.append(feature)        #Agregamos el feature a cada elemento de la lista de result...
				else:
					if "*" not in i:
						i.append("*")

			result.append([])
		elif(root.tag=="paralel"):
			#l = set()
			for i in list(root):
				compute = self.compute_denotational(i, featureP)
				#if compute not in val:
				if result == []:
					result = compute
					self.reduce(result)
				else:
					varx = []#list(result)
					k = 0
					self.reduce(result)
					self.reduce(compute)
					#print len(result)
		            #print len(compute)
					for i in result:
						for j in compute:
							#if j not in result:
							#print k
							k = k+1
							new = list(i)
							new.extend(j)                          
							varx.append(new)
					if varx != []:
						result = varx
					lst2=[]
					[lst2.append(key) for key in result if key not in lst2]
					result = lst2
		elif(root.tag=="choose_1"):
			for i in list(root):
				resaux = self.compute_denotational(i, featureP)
				for j in resaux:
					result.append(j)
			m = map(frozenset,result)
			result = list(map(list,m))
		elif(root.tag =="requires"):
			#print "Requires"
#			feat1 = root.get("feature_1")
#			feat2 = root.get("feature_2")
			for i in list(root):
				result = self.compute_denotational(i, featureP)
#				resaux = self.compute_denotational(i, featureP)
#				for j in resaux:
#					if feat1 in j and feat2 not in j:
#						j.append(feat2)
#					result.append(j)
#			if self.prev != [] and self.prev.sort() == result.sort():
#				self.add_warning(feat1 + " requires " + feat2,self.prev,result)
#			self.prev = list(result)
		elif (root.tag=="excludes"):
			#print "Excludes"
#			feat1 = root.get("feature_1")
#			feat2 = root.get("feature_2")
#			self.excludes_list
#			self.excludes_list.append((feat1,feat2))
			for i in list(root):
				result = self.compute_denotational(i, featureP)
#				resaux = self.compute_denotational(i, featureP)
#				for j in resaux:
#					if feat1 not in j or feat2 not in j:
#						result.append(j)
#			if self.prev != [] and self.prev.sort() == result.sort():
#				self.add_warning(feat1+" excludes "+feat2,self.prev,result)
#			self.prev = list(result)
		if (result == []):
			self.debug("Error tag {0} not processed".format(root.tag))

		#m = set(map(frozenset,result))
		#return list(map(list,m))
		return result

#	def reduce(self, l):
#		self.excludes_list
#		r = []
#		for (feat1,feat2) in self.excludes_list:
#			for j in l:
#				if feat1 not in j or feat2 not in j:
#					r.append(j)
#		l = list(r)

	def getProb(self, featureP):
		"""
		Run function
		"""
		
		xml = ET.parse(self.file)
		results = self.compute_denotational(xml.getroot(), featureP)
		a =float(sum(x.count(featureP) for x in results[0]))
		b =float(len(results[0]))
		return float(a/b)
    
	def getFeatures(self):
		return self.features
    
	def getProbs(self):
		return self.probs
	
	def getRoot(self):
		return self.root
	
	def getChildren(self, feature):
		res = self.node_to_children.get(feature)
		if res:
			return res
		else:
			return []

class ant_colony:
	class ant(Thread):
		def __init__(self, sp, nodes, init_location, possible_locations, value_to_id,
			   pheromone_map, distance_callback, alpha, beta, first_pass=False):
			"""
			initialized an ant, to traverse the map
			init_location -> marks where in the map that the ant starts
			possible_locations -> a list of possible nodes the ant can go to
				when used internally, gives a list of possible locations the ant can traverse to _minus those nodes already visited_
			pheromone_map -> map of pheromone values for each traversal between each node
			distance_callback -> is a function to calculate the distance between two nodes
			alpha -> a parameter from the ACO algorithm
				to control the influence of the amount of pheromone when making a choice in _pick_path()
			beta -> a parameters from ACO that controls the influence of the distance to the next node in _pick_path()
			first_pass -> if this is a first pass on a map, then do some steps differently, noted in methods below
			
			route -> a list that is updated with the labels of the nodes that the ant has traversed
			pheromone_trail -> a list of pheromone amounts deposited along the ants trail, maps to each traversal in route
			distance_traveled -> total distance tranveled along the steps in route
			location -> marks where the ant currently is
			tour_complete -> flag to indicate the ant has completed its traversal
				used by get_route() and get_distance_traveled()
			"""
			Thread.__init__(self)
			
			self.sp = sp
			self.nodes = nodes
			self.init_location = init_location
			self.possible_locations = possible_locations	
			self.value_to_id = value_to_id
			self.route = []
			self.distance_traveled = self.sp.getProb(self.nodes.get(self.init_location))
			self.location = init_location
			self.pheromone_map = pheromone_map
			self.distance_callback = distance_callback
			self.alpha = alpha
			self.beta = beta
			self.first_pass = first_pass
			
			#append start location to route, before doing random walk
			self._update_route(init_location)
			
			self.tour_complete = False
			
		def run(self):
			"""
			until self.possible_locations is empty (the ant has visited all nodes)
				_pick_path() to find a next node to traverse to
				_traverse() to:
					_update_route() (to show latest traversal)
					_update_distance_traveled() (after traversal)
			return the ants route and its distance, for use in ant_colony:
				do pheromone updates
				check for new possible optimal solution with this ants latest tour
			"""
			while self.possible_locations and not self.tour_complete:
				nex = self._pick_path()
				if nex:
					self._traverse(self.location, nex)
				else:
					self.tour_complete = True
			self.tour_complete = True
		
		def _pick_path(self):
			"""
			source: https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms#Edge_selection
			implements the path selection algorithm of ACO
			calculate the attractiveness of each possible transition from the current location
			then randomly choose a next path, based on its attractiveness
			"""
			#on the first pass (no pheromones), then we can just choice() to find the next one
			if self.first_pass:
				import random
				return random.choice(self.possible_locations)
			
			if self.sp.getFeatures().get(self.nodes.get(self.location)):
				import random
				if random.choice([True, False]):
					return None
			
			attractiveness = dict()
			sum_total = 0.0
			#for each possible location, find its attractiveness (it's (pheromone amount)*1/distance [tau*eta, from the algortihm])
			#sum all attrativeness amounts for calculating probability of each route in the next step
			for possible_next_location in self.possible_locations:
				#NOTE: do all calculations as float, otherwise we get integer division at times for really hard to track down bugs
				pheromone_amount = float(self.pheromone_map[self.location][possible_next_location])
				distance = float(self.distance_callback(self.location, possible_next_location))
				
				#tau^alpha * eta^beta
				#TODO: change 1/dist for 1+dist?
				attractiveness[possible_next_location] = float(pow(pheromone_amount, self.alpha)*pow(distance, self.beta))
				sum_total += attractiveness[possible_next_location]
			
			#it is possible to have small values for pheromone amount / distance, such that with rounding errors this is equal to zero
			#rare, but handle when it happens
			if sum_total == 0.0:
				#increment all zero's, such that they are the smallest non-zero values supported by the system
				#source: http://stackoverflow.com/a/10426033/5343977
				def next_up(x):
					import math
					import struct
					# NaNs and positive infinity map to themselves.
					if math.isnan(x) or (math.isinf(x) and x > 0):
						return x

					# 0.0 and -0.0 both map to the smallest +ve float.
					if x == 0.0:
						x = 0.0

					n = struct.unpack('<q', struct.pack('<d', x))[0]
					
					if n >= 0:
						n += 1
					else:
						n -= 1
					return struct.unpack('<d', struct.pack('<q', n))[0]
					
				for key in attractiveness:
					attractiveness[key] = next_up(attractiveness[key])
				sum_total = next_up(sum_total)
			
			#cumulative probability behavior, inspired by: http://stackoverflow.com/a/3679747/5343977
			#randomly choose the next path
			import random
			toss = random.random()
					
			cummulative = 0
			for possible_next_location in attractiveness:
				weight = (attractiveness[possible_next_location] / sum_total) #TODO: change / for * ?
				if toss <= weight + cummulative:
					return possible_next_location
				cummulative += weight
		
		def _traverse(self, start, end):
			"""
			_update_route() to show new traversal
			_update_distance_traveled() to record new distance traveled
			self.location update to new location
			called from run()
			"""
			self._update_route(end)
			self._update_distance_traveled(start, end)
			self.location = end
		
		def _update_route(self, new):
			"""
			add new node to self.route
			remove new node form self.possible_location
			called from _traverse() & __init__()
			"""
			self.route.append(new)
			self.possible_locations = [self.value_to_id.get(x) for x in self.sp.getChildren(self.nodes.get(new))]
			
		def _update_distance_traveled(self, start, end):
			"""
			use self.distance_callback to update self.distance_traveled
			"""
			self.distance_traveled *= float(self.distance_callback(start, end)) #TODO: change + for * ?
	
		def get_route(self):
			if self.tour_complete:
				return self.route
			return None
			
		def get_distance_traveled(self):
			if self.tour_complete:
				return self.distance_traveled
			return None
		
	def __init__(self, sp, start=None, ant_count=5, alpha=.5, beta=1.2,
			  pheromone_evaporation_coefficient=.40, pheromone_constant=1000.0, iterations=10):
		"""
		initializes an ant colony (houses a number of worker ants that will traverse a map
							 to find an optimal route as per ACO [Ant Colony Optimization])
		source: https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms
		
        splap -> is assumed to be the class that will give the probabilities of the features
			as well as the possible next nodes
			
		start -> if set, then is assumed to be the node where all ants start their traversal
			if unset, then assumed to be the first key of nodes when sorted()
			
		alpha -> a parameter from the ACO algorithm to control the influence of the amount of pheromone when an ant makes a choice
		
		beta -> a parameters from ACO that controls the influence of the distance to the next node in ant choice making
			
		pheromone_evaporation_coefficient -> a parameter used in removing pheromone values from the pheromone_map (rho in ACO algorithm)
			used by _update_pheromone_map()
		
		pheromone_constant -> a parameter used in depositing pheromones on the map (Q in ACO algorithm)
			used by _update_pheromone_map()
		
		iterations -> how many iterations to let the ants traverse the map
        
		nodes -> is assumed to be a list of all the features of the expression
			
		distance_callback -> is assumed to take a pair of coordinates and return the distance between them
			populated into distance_matrix on each call to get_distance()
		
		distance_matrix -> holds values of distances calculated between nodes
			populated on demand by _get_distance()
		
		pheromone_map -> holds final values of pheromones
			used by ants to determine traversals
			pheromone dissipation happens to these values first, before adding pheromone values from the ants during their traversal
			(in ant_updated_pheromone_map)
			
		ant_updated_pheromone_map -> a matrix to hold the pheromone values that the ants lay down
			not used to dissipate, values from here are added to pheromone_map after dissipation step
			(reset for each traversal)
		
		ants -> holds worker ants
			they traverse the map as per ACO
			notable properties:
				total distance traveled
				route
			
		first_pass -> flags a first pass for the ants, which triggers unique behavior
		
		shortest_distance -> the shortest distance seen from an ant traversal
		
		shortets_path_seen -> the shortest path seen from a traversal (shortest_distance is the distance along this path)
		"""
        
		#splap
		self.sp = sp
        
		#nodes
		self.nodes = list(self.sp.getFeatures().keys())
		
		#create internal mapping and mapping for return to caller
		self.id_to_value, self.value_to_id = self._init_nodes(self.nodes)
		#create matrix to hold distance calculations between nodes
		self.distance_matrix = self._init_matrix(len(self.nodes))
		#create matrix for master pheromone map, that records pheromone amounts along routes
		self.pheromone_map = self._init_matrix(len(self.nodes))
		#create a matrix for ants to add their pheromones to, before adding those to pheromone_map during the update_pheromone_map step
		self.ant_updated_pheromone_map = self._init_matrix(len(self.nodes))
		#dict for the already computed probabilities
		self.feature_to_prob = dict()
		
		#distance_callback
		def distance_callback(sp, ftp, f1, f2):
			if f2 in sp.getChildren(f1):
				pf1 = 0.0
				pf2 = 0.0
				if f1 in ftp:
					pf1 = ftp.get(f1)
				else:
					pf1 = sp.getProb(f1)
					ftp[f1] = pf1
				if f2 in ftp:
					pf2 = ftp.get(f2)
				else:
					pf2 = sp.getProb(f2)
					ftp[f2] = pf2
				if pf1 == 0 or pf2 == 0 or pf1 > 1 or pf2 > 1 or pf2 > pf1:
					print(str(pf1) + " --> " + str(f1) + " ; " + str(pf2) + " --> " + str(f2))
				return float(pf2/pf1)
			else:
				return -1
			
		self.distance_callback = distance_callback
		
		#start
		if start is None:
			self.start = 0
		else:
			self.start = None
			#init start to internal id of node id passed
			for key, value in self.id_to_value.items():
				if value == start:
					self.start = key
			
			#if we didn't find a key in the nodes passed in, then raise
			if self.start is None:
				raise KeyError("Key: " + str(start) + " not found in the nodes dict passed.")
		
		#ant_count
		if type(ant_count) is not int:
			raise TypeError("ant_count must be int")
			
		if ant_count < 1:
			raise ValueError("ant_count must be >= 1")
		
		self.ant_count = ant_count
		
		#alpha	
		if (type(alpha) is not int) and type(alpha) is not float:
			raise TypeError("alpha must be int or float")
		
		if alpha < 0:
			raise ValueError("alpha must be >= 0")
		
		self.alpha = float(alpha)
		
		#beta
		if (type(beta) is not int) and type(beta) is not float:
			raise TypeError("beta must be int or float")
			
		if beta < 1:
			raise ValueError("beta must be >= 1")
			
		self.beta = float(beta)
		
		#pheromone_evaporation_coefficient
		if (type(pheromone_evaporation_coefficient) is not int) and type(pheromone_evaporation_coefficient) is not float:
			raise TypeError("pheromone_evaporation_coefficient must be int or float")
		
		self.pheromone_evaporation_coefficient = float(pheromone_evaporation_coefficient)
		
		#pheromone_constant
		if (type(pheromone_constant) is not int) and type(pheromone_constant) is not float:
			raise TypeError("pheromone_constant must be int or float")
		
		self.pheromone_constant = float(pheromone_constant)
		
		#iterations
		if (type(iterations) is not int):
			raise TypeError("iterations must be int")
		
		if iterations < 0:
			raise ValueError("iterations must be >= 0")
			
		self.iterations = iterations
		
		#other internal variable init
		self.first_pass = True
		self.ants = self._init_ants(self.start)
		self.shortest_distance = None
		self.shortest_path_seen = None
		
	def _get_distance(self, start, end):
		"""
		uses the distance_callback to return the distance between nodes
		if a distance has not been calculated before, then it is populated in distance_matrix and returned
		if a distance has been called before, then its value is returned from distance_matrix
		"""
		if not self.distance_matrix[start][end]:
			distance = self.distance_callback(self.sp, self.feature_to_prob, self.id_to_value[start], self.id_to_value[end])
			
			if (type(distance) is not int) and (type(distance) is not float):
				raise TypeError("distance_callback should return either int or float, saw: "+ str(type(distance)))
			
			self.distance_matrix[start][end] = float(distance)
			return distance
		return self.distance_matrix[start][end]
		
	def _init_nodes(self, nodes):
		"""
		create a mapping of internal id numbers (0 .. n) to the keys in the nodes passed 
		create a mapping of the id's to the values of nodes
		we use id_to_key to return the route in the node names the caller expects in mainloop()
		"""
		id_to_value = dict()
		value_to_id = dict()
		
		for id in range(len(nodes)):
			id_to_value[id] = nodes[id]
			value_to_id[nodes[id]] = id
			
		return id_to_value, value_to_id
		
	def _init_matrix(self, size, value=0.0):
		"""
		setup a matrix NxN (where n = size)
		used in both self.distance_matrix and self.pheromone_map
		as they require identical matrixes besides which value to initialize to
		"""
		ret = []
		for row in range(size):
			ret.append([float(value) for x in range(size)])
		return ret
	
	def _init_ants(self, start):
		"""
		on first pass:
			create a number of ant objects
		on subsequent passes, just call __init__ on each to reset them
		by default, all ants start at the first node, 0
		as per problem description: https://www.codeeval.com/open_challenges/90/
		"""
		#allocate new ants on the first pass
		if self.first_pass:
			return [self.ant(self.sp, self.id_to_value, start,
				[self.value_to_id.get(x) for x in self.sp.getChildren(self.id_to_value.get(start))],
				self.value_to_id, self.pheromone_map, self._get_distance, self.alpha, self.beta, first_pass=True)
				for x in range(self.ant_count)]
		#else, just reset them to use on another pass
		for ant in self.ants:
			ant.__init__(self.sp, self.id_to_value, start,
				[self.value_to_id.get(x) for x in self.sp.getChildren(self.id_to_value.get(start))],
				self.value_to_id, self.pheromone_map, self._get_distance, self.alpha, self.beta)
	
	def _update_pheromone_map(self):
		"""
		1)	Update self.pheromone_map by decaying values contained therein via the ACO algorithm
		2)	Add pheromone_values from all ants from ant_updated_pheromone_map
		called by:
			mainloop()
			(after all ants have traveresed)
		"""
		#always a square matrix
		for start in range(len(self.pheromone_map)):
			for end in range(len(self.pheromone_map)):
				#decay the pheromone value at this location
				#tau_xy <- (1-rho)*tau_xy	(ACO)
				self.pheromone_map[start][end] = (1-self.pheromone_evaporation_coefficient)*self.pheromone_map[start][end]
				
				#then add all contributions to this location for each ant that travered it
				#(ACO)
				#tau_xy <- tau_xy + delta tau_xy_k
				#	delta tau_xy_k = Q / L_k
				self.pheromone_map[start][end] += self.ant_updated_pheromone_map[start][end]
	
	def _populate_ant_updated_pheromone_map(self, ant):
		"""
		given an ant, populate ant_updated_pheromone_map with pheromone values according to ACO
		along the ant's route
		called from:
			mainloop()
			( before _update_pheromone_map() )
		"""
		route = ant.get_route()
		for i in range(len(route)-1):
			#find the pheromone over the route the ant traversed
			current_pheromone_value = float(self.ant_updated_pheromone_map[route[i]][route[i+1]])
		
			#update the pheromone along that section of the route
			#(ACO)
			#	delta tau_xy_k = Q / L_k
			new_pheromone_value = self.pheromone_constant*ant.get_distance_traveled() #TODO: change / for * ?
			
			self.ant_updated_pheromone_map[route[i]][route[i+1]] = current_pheromone_value + new_pheromone_value
			self.ant_updated_pheromone_map[route[i+1]][route[i]] = current_pheromone_value + new_pheromone_value
		
	def mainloop(self):
		"""
		Runs the worker ants, collects their returns and updates the pheromone map with pheromone values from workers
			calls:
			_update_pheromones()
			ant.run()
		runs the simulation self.iterations times
		"""
		
		for _ in range(self.iterations):
			#start the multi-threaded ants, calls ant.run() in a new thread
			for ant in self.ants:
				ant.start()
			
			#source: http://stackoverflow.com/a/11968818/5343977
			#wait until the ants are finished, before moving on to modifying shared resources
			for ant in self.ants:
				ant.join()
			
			for ant in self.ants:	
				#update ant_updated_pheromone_map with this ant's constribution of pheromones along its route
				self._populate_ant_updated_pheromone_map(ant)
				
				#if we haven't seen any paths yet, then populate for comparisons later
				if not self.shortest_distance:
					self.shortest_distance = ant.get_distance_traveled()
				
				if not self.shortest_path_seen:
					self.shortest_path_seen = ant.get_route()
					
				#if we see a shorter path, then save for return
				if ant.get_distance_traveled() > self.shortest_distance:
					self.shortest_distance = ant.get_distance_traveled()
					self.shortest_path_seen = ant.get_route()
			
			#decay current pheromone values and add all pheromone values we saw during traversal (from ant_updated_pheromone_map)
			self._update_pheromone_map()
			
			#flag that we finished the first pass of the ants traversal
			if self.first_pass:
				self.first_pass = False
			
			#reset all ants to default for the next iteration
			self._init_ants(self.start)
			
			#reset ant_updated_pheromone_map to record pheromones for ants on next pass
			self.ant_updated_pheromone_map = self._init_matrix(len(self.nodes), value=0)
		
		#translate shortest path back into callers node id's
		ret = []
		for id in self.shortest_path_seen:
			ret.append(self.id_to_value[id])
		
		return ret, self.shortest_distance

def loop(source):
	"""
	Loop function
	"""
	features = {}
	start = ""
	tree = ET.parse(source).getroot()
	node_to_children = dict()
	for featureP in list(tree.iter()):
		if featureP.tag == 'mandatory_feature' or featureP.tag == 'optional_feature':
			features[featureP.get('name')] = True in [x.tag == 'checkmark' for x in list(featureP)]
			lis = []
			remanent = list(featureP)
			while remanent:
				f = remanent.pop()
				if f.get('name', None):
					lis.append(f)
				else:
					for fc in list(f):
						remanent.append(fc)
			if lis:
				node_to_children[featureP.get('name')] = [x.get('name') for x in lis]
			if not start:
				start = featureP.get('name')
    
	start_time = time.time()
	sp = splap(source, features, node_to_children)
	end_time = time.time()
	brute_force_time = end_time - start_time
#	print("Brute Force time: " + str(brute_force_time))
	
	start_time = time.time()
	aco = ant_colony(sp, start=start)
	best, dist = aco.mainloop()
	end_time = time.time()
	aco_time = end_time - start_time
	
	probs = sp.getProbs()
	valid_probs = dict()
	for k in probs.keys():
		if features.get(k):
			valid_probs[k] = probs.get(k)
	
	bfr = max(valid_probs.items(), key=operator.itemgetter(1))[0]
	acor = best
	bfp = max(valid_probs.values())
	acop = dist
	bft = brute_force_time
	acot = aco_time
	
	print("Source: " + source)
	print("======================================")
	print("Brute Force route: " + str(bfr))
	print("ACO route: " + str(acor))
	print("Brute Force prob: " + str(bfp))
	print("ACO prob: " + str(acop))
	print("Brute Force time: " + str(bft))
	print("ACO time: " + str(acot))
	print("======================================")
	
	return bfr, acor, bfp, acop, bft, acot

def main():
	"""
	Main function
	"""
	ps = []
	ts = []
	meanPs = 0.0
	meanTs = 0.0
	outP = open("probs.txt", 'w')
	outT = open("times.txt", 'w')
	outP.write("Iter | Brute Force | ACO | Increment\n")
	outT.write("Iter | Brute Force | ACO | Derement\n")
	outP.flush()
	outT.flush()
	print("======================================")
	onlyfiles = [f for f in listdir('exps/') if isfile(join('exps/', f))]
	i = 1
	for fai in onlyfiles:
		if fai.endswith(".xml"):
			source = 'exps/'+fai
			bfr, acor, bfp, acop, bft, acot = loop(source)
			ps.append([bfp, acop, 1.0 - float(acop/bfp)])
			ts.append([bft, acot, 1.0 - float(acot/bft)])
			outP.write(str(i) + " & " + str(bfp) + " & " + str(acop) + " & " + str(1.0 - float(acop/bfp)) + "\\\\\n")
			outP.write("\hline\n")
			outP.flush()
			outT.write(str(i) + " & " + str(bft) + " & " + str(acot) + " & " + str(1.0 - float(acot/bft)) + "\\\\\n")
			outT.write("\hline\n")
			outT.flush()
			meanPs += 1.0 - float(acop/bfp)
			meanTs += 1.0 - float(acot/bft)
			i += 1
	outP.close()
	outT.close()
	print("MeanPs = " + str(float(meanPs/i)))
	print("MeanT`s = " + str(float(meanTs/i)))
	
	x = np.array(range(1, len(ts) + 1))
	bfps = []
	acops = []
	bfts = []
	acots = []
	incs = []
	decs = []
	for i in sorted(ps, reverse=True):
		bfps.append(i[0])
		acops.append(i[1])
		incs.append(i[2])
	for i in sorted(ts):
		bfts.append(i[0])
		acots.append(i[1])
		decs.append(i[2])
	
	fig, ax = plt.subplots()
	ax.scatter(x, bfps, color='blue', s=5, label="measured")
	ax.scatter(x, acops, color='red', s=5, label="measured")
	plt.axis([-1, len(bfps) + 1, -0.1, 1.1])  # Put in side your range [xmin,xmax,ymin,ymax], like ax.axis([-5,5,-5,200])
	plt.xlabel('Product Line number')
	plt.ylabel('Product Line best probability')
	plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
#	plt.legend(['Brute Force', 'ACO'], loc='upper rigth')
	plt.savefig('probs.eps', format='eps')
	fig, ax = plt.subplots()
	ax.scatter(x, bfts, color='blue', s=5, label="measured")
	ax.scatter(x, acots, color='red', s=5, label="measured")
	plt.axis([-1, len(bfts) + 1, -0.1, max([max(bfts), max(acots)]) + 1])  # Put in side your range [xmin,xmax,ymin,ymax], like ax.axis([-5,5,-5,200])
	plt.xlabel('Product Line number')
	plt.ylabel('Product Line best time')
	plt.savefig('times.eps', format='eps')
	fig, ax = plt.subplots()
	ax.scatter(x, sorted(incs), color='green', s=5, label="measured")
	plt.axis([-1, len(bfps) + 1, -0.1, 1.1])  # Put in side your range [xmin,xmax,ymin,ymax], like ax.axis([-5,5,-5,200])
	plt.xlabel('Product Line number')
	plt.ylabel('Product Line probability loss')
	plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
#	plt.legend(['Brute Force', 'ACO'], loc='upper rigth')
	plt.savefig('probDiff.eps', format='eps')
	fig, ax = plt.subplots()
	ax.scatter(x, sorted(decs), color='green', s=5, label="measured")
	plt.axhline(color='black')
	plt.axis([-1, len(bfts) + 1, min(decs)-0.1, max(decs)+0.1])  # Put in side your range [xmin,xmax,ymin,ymax], like ax.axis([-5,5,-5,200])
	plt.xlabel('Product Line number')
	plt.ylabel('Product Line time saving')
	plt.savefig('timeDiff.eps', format='eps')
#	fig, ax = plt.subplots()
#	ax.scatter(x, bfps, color='blue', s=5, label="measured")
#	ax.scatter(x, acops, color='red', s=5, label="measured")
#	ax.scatter(x, incs, color='green', s=5, label="measured")
#	plt.axis([-1, len(bfps) + 1, 0, 1])  # Put in side your range [xmin,xmax,ymin,ymax], like ax.axis([-5,5,-5,200])
#	plt.xlabel('Product Line number')
#	plt.ylabel('Product Line best probability')
#	plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
##	plt.legend(['Brute Force', 'ACO'], loc='upper rigth')
#	plt.savefig('probsDiff.eps', format='eps')
#	fig, ax = plt.subplots()
#	ax.scatter(x, bfts, color='blue', s=5, label="measured")
#	ax.scatter(x, acots, color='red', s=5, label="measured")
#	ax.scatter(x, decs, color='green', s=5, label="measured")
#	plt.axis([-1, len(bfts) + 1, 0, max([max(bfts), max(acots)]) + 1])  # Put in side your range [xmin,xmax,ymin,ymax], like ax.axis([-5,5,-5,200])
#	plt.xlabel('Product Line number')
#	plt.ylabel('Product Line best time')
#	plt.savefig('timesDiff.eps', format='eps')

if __name__ == "__main__":
#	loop("prenornal_foda05700.xml")
	main()