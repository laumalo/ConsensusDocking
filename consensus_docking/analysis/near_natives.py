import logging
import sys 
import collections

logging.basicConfig(format="%(message)s", level=logging.INFO,
                    stream=sys.stdout)

class NearNatives(object):
	"""
	It contains all the methods used to analyze near native structures during 
	the clustering. 
	"""
	def __init__(self):
		pass

	def _get_near_natives(self, rmsd_file, threshold):
		"""
		It gets structures of a rmsd results file that are below a certain
		threscold RMSD (in A).

		Parameters
		----------
		rmsd_path : str
			Path to a .csv file containing the RMSD of each structure. 
		threshold : float
			RMSD value (in A) below a structures is considered near native.
		"""
		with open(rmsd_file, 'r') as f: 
			# Parse RMSD file
			d = {line.split()[0]: float(line.split()[1]) 
			     for line in f.readlines()}
			d_sorted = dict(sorted(d.items(), key=lambda item: item[1]))
		
		# Extract near native structures in the set
		near_native_structures = \
			[os.basename(k) for k in list(d_sorted) if d[k]<=threshold]
		return near_native_structures

	@static_method
	def extract_near_natives(rmsd_folder, output_file = None, threshold = 5):
		"""
		It gets all the structures that are below a certain threscold RMSD 
		(in A).

		Parameters
		----------
		rmsd_folder : str
			Path to folder containing all the RMSD information. 
		output_file : str
			Path to write out the list of near native structures, if it is None 
			it does not generate any output file. Default: None
		threshold : float
			RMSD value (in A) below a structures is considered near native.
			Default: 5.
		"""
		files = [f for f in os.listdir(rmsd_folder) if f.startswith('rmsd_')]
		near_native_structures = []
		
		# Extract all near natives structures
		for file in files: 
			structures = self.get_near_natives(file)
			near_native_structures = \
				np.concatenate(near_native_structures, structures)
		
		# Save output file with list of near natives
		if output_file is not None:
			with open(output_file, 'w') as f:
			    for structure in near_native_structures:
			        f.write("%s\n" % structure)
		return near_native_structures

	@static_method
	def near_natives_in_clusters(near_native_structures, cluster_dict):
		"""
		It counts how many near native structures are in each cluster. 

		Parameters
		----------
		near_native_structures : list
			List of near native structures
		cluster_dict : dict
			Dictionary of the clusters.
		"""
		nn_in_cluster = []
		for pose in near_native_structures:
		    try:
		        nn_in_cluster.append(cluster_dict[pose])
		    except Exception: 
		        continue

		logging.info(collections.Counter(map(int,nn_in_cluster)))




