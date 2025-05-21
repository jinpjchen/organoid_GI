import pandas as pd
import numpy as np
from scipy.io import mmread
import os
import six
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
from time import time
from tqdm import tqdm_notebook as progress
from collections import defaultdict
import warnings
from .util import _strip_cat_cols

# CellPopulation class
class CellPopulation:
	
	def __init__(self, matrix, cell_list, gene_list, source='arrays', normalized_matrix=None, calculate_statistics=True):
		"""A class for holding single-cell data with separate tables for expression, normalized expression, and data about cells and genes
		
		Args:
		
		matrix: expression matrix of UMIs per cells (rows = cell barcodes, columns = Ensembl gene ids)
		
		cell_list: table of properties of cells in the population (indexed by cell barcode)
		
		gene_list: table of properties of genes in the population (indexed by Ensembl gene id)
		
		source: keeps track of how this population was derived. If it is from raw data, the source will be 'arrays'. If it is a subpopulation derived from another population, this will contain the list of criteria that were used to create that subpopulation
				
		normalized_matrix: Add the supplied normalized expression matrix
		"""
		self.matrix = matrix
		self.normalized_matrix = normalized_matrix
		
		if calculate_statistics:
			# fill out the list of gene properties
			print("Generating summary statistics...")
			gene_list['mean'] = matrix.mean()
			gene_list['std'] = matrix.std()
			gene_list['cv'] = gene_list['std']/gene_list['mean']
			gene_list['fano'] = gene_list['std']**2/gene_list['mean']
			gene_list['in_matrix'] = True
		self.genes = gene_list
		
		self.cells = cell_list
		
		if calculate_statistics:
			self.cells['gem_group'] = self.cells.index.map(lambda x: int(x.split('-')[-1]))
			self.guides = sorted(cell_list[cell_list['single_cell']]['guide_identity'].unique())
		
		self.source = source
				
		print("Done.")

	@classmethod
	def from_hdf(cls, filename):
		t = time()
		
		with pd.HDFStore(filename) as store:
			print('Loading matrix...')
			matrix = store['matrix']
			if '/normalized_matrix' in store.keys():
				print('Loading normalized matrix...')
				normalized_matrix = store['normalized_matrix']
			else:
				normalized_matrix = None
			print('Loading metadata...')
			gene_list = store['gene_list']
			cell_list = store['cell_list']
		
		print('Done in {0}s.'.format(time() - t))    
		
		return cls(matrix, cell_list, gene_list, source=filename, normalized_matrix=normalized_matrix, calculate_statistics=False)
	
	def to_hdf(self, filename, store_normalized_matrix=False):
		t = time()
		
		with pd.HDFStore(filename) as store:
			print('Writing matrix...')
			store.put('matrix', self.matrix)
			if store_normalized_matrix and self.normalized_matrix is not None:
				print('Writing normalized matrix...')
				store.put('normalized_matrix', self.normalized_matrix)
			print('Writing metadata...')
			# hack because HDFStore complains about categorical columns
			store.put('cell_list', _strip_cat_cols(self.cells))
			store.put('gene_list', _strip_cat_cols(self.genes))
			
		print('Done in {0}s.'.format(time() - t))
		
	@classmethod
	def from_file(cls, directory, genome=None, filtered=True, raw_umi_threshold=2000, version='v3', direct_capture=False):
		"""Load a Perturb-seq data set
		Example:
		pop = CellPopulation.from_file('~/sequencing/perturbseq_expt/', genome='GRCh38')
		
		Args:
		
		directory: location of base path of 10x experiment. The filtered gene-barcode matrices will automatically be found
			
		genome: name of genome reference you aligned to (e.g. GRCh38)
		"""
		if genome is None:
			genome = 'GRCh38'
		
		if version == 'v2':
						# output of cellranger count and cellranger aggr has slightly different structure
			if os.path.isdir(os.path.join(directory, os.path.normpath("outs/filtered_gene_bc_matrices"), genome)):
				if filtered:
					matrix_directory = os.path.join(directory, os.path.normpath("outs/filtered_gene_bc_matrices"), genome)
				else:
					matrix_directory = os.path.join(directory, os.path.normpath("outs/raw_gene_bc_matrices"), genome)
			else:
				if filtered:
					matrix_directory = os.path.join(directory, os.path.normpath("outs/filtered_gene_bc_matrices_mex"), genome)
				else:
					matrix_directory = os.path.join(directory, os.path.normpath("outs/raw_gene_bc_matrices_mex"), genome)
			
			print('Loading digital expression data: {0}...'.format(os.path.join(matrix_directory, "matrix.mtx")))
			genes_path = os.path.join(matrix_directory, "genes.tsv")
			gene_list = pd.read_csv(genes_path, sep='\t', header=None, names=['gene_id', 'gene_name'])
			
			barcodes_path = os.path.join(matrix_directory, "barcodes.tsv")
			cell_barcodes = pd.read_csv(barcodes_path, sep='\t', header=None, names=['cell_barcode'])
			# for the moment we keep matrix as dense... should be sparse
			matrix = mmread(os.path.join(matrix_directory, "matrix.mtx"))
			
			print('Densifying matrix...')
			if filtered:
				matrix = pd.DataFrame(matrix.transpose().todense(),
									  columns=gene_list['gene_id'],
									  index=cell_barcodes['cell_barcode'],
									  dtype='int32')
			else:
				# raw gene-barcode matrices generally too big to fit in memory so must do some thresholding
				m = pd.Series(np.asarray(np.sum(matrix, axis=0)).flatten())
				ind = m[m >= raw_umi_threshold].index.values
				print('Filtering cell barcodes with fewer than {0} UMIs...'.format(raw_umi_threshold))
				matrix = matrix.tocsc()[:, ind]
				matrix = pd.DataFrame(matrix.transpose().todense(),
									  columns=gene_list['gene_id'],
									  index=cell_barcodes['cell_barcode'].iloc[ind],
									  dtype='int32')
				
			# table of gene properties
			gene_list = gene_list.set_index('gene_id')
			
			if filtered:
				identity_filename = 'cell_identities.csv'
			else:
				identity_filename = 'raw_cell_identities.csv'
			
			print("Loading guide identities:" + os.path.join(directory, 'outs', identity_filename) + '...')
			# adjust for historical differences in column names...
			guide_identities = pd.read_csv(os.path.join(directory, 'outs', identity_filename)).rename(columns={'cell BC': 'cell_barcode',
			'read count': 'guide_read_count',
			'UMI count': 'guide_UMI_count',
			'coverage': 'guide_coverage',
			'cell_BC': 'cell_barcode',
			'read_count': 'guide_read_count',
			'UMI_count': 'guide_UMI_count'}).set_index('cell_barcode')
			cols = guide_identities.columns
			cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, six.string_types) else x)
			guide_identities.columns = cols
					
			# table of cell properties
			if filtered:
				cell_list = pd.merge(cell_barcodes, guide_identities, left_on='cell_barcode', right_index=True, how='left').set_index('cell_barcode')
			else:
				cell_list = pd.merge(cell_barcodes.iloc[ind], guide_identities, left_on='cell_barcode', right_index=True, how='left').set_index('cell_barcode')
			guide_targets = cell_list['guide_identity'].map(lambda x: str(x).split('_')[0])
			guide_targets.name = 'guide_target'
			cell_list['guide_target'] = guide_targets
			
			cell_list['single_cell'] = (cell_list['number_of_cells'] == 1) & (cell_list['good_coverage']) & (~(cell_list['guide_identity'] == '*'))
			cell_list['UMI_count'] = matrix.sum(axis=1)      

		else:
			# output of cellranger count and cellranger aggr has slightly different structure
			if os.path.isdir(os.path.join(directory, os.path.normpath("outs/filtered_feature_bc_matrix"))):
				if filtered:
					matrix_directory = os.path.join(directory, os.path.normpath("outs/filtered_feature_bc_matrix"))
				else:
					matrix_directory = os.path.join(directory, os.path.normpath("outs/raw_feature_bc_matrix"))
			# else:
			#     if filtered:
			#         matrix_directory = os.path.join(directory, os.path.normpath("outs/filtered_gene_bc_matrices_mex"), genome)
			#     else:
			#         matrix_directory = os.path.join(directory, os.path.normpath("outs/raw_gene_bc_matrices_mex"), genome)
			
			print('Loading digital expression data: {0}...'.format(os.path.join(matrix_directory, "matrix.mtx.gz")))
			genes_path = os.path.join(matrix_directory, "features.tsv.gz")
			gene_list = pd.read_csv(genes_path, sep='\t', header=None, names=['gene_id', 'gene_name', 'feature_type'])

			barcodes_path = os.path.join(matrix_directory, "barcodes.tsv.gz")
			cell_barcodes = pd.read_csv(barcodes_path, sep='\t', header=None, names=['cell_barcode'])
			# for the moment we keep matrix as dense... should be sparse
			matrix = mmread(os.path.join(matrix_directory, "matrix.mtx.gz"))
			
			print('Densifying matrix...')
			if filtered:
				matrix = pd.DataFrame(matrix.transpose().todense(),
									  columns=gene_list['gene_id'],
									  index=cell_barcodes['cell_barcode'],
									  dtype='int32')
			else:
				# raw gene-barcode matrices generally too big to fit in memory so must do some thresholding
				m = pd.Series(np.asarray(np.sum(matrix, axis=0)).flatten())
				ind = m[m >= raw_umi_threshold].index.values
				print('Filtering cell barcodes with fewer than {0} UMIs...'.format(raw_umi_threshold))
				matrix = matrix.tocsc()[:, ind]
				matrix = pd.DataFrame(matrix.transpose().todense(),
									  columns=gene_list['gene_id'],
									  index=cell_barcodes['cell_barcode'].iloc[ind],
									  dtype='int32')

			gene_list = gene_list.set_index('gene_id')

			if direct_capture:
				guide_list = gene_list[gene_list["feature_type"] == "CRISPR Guide Capture"]
				gene_list = gene_list[gene_list["feature_type"] == "Gene Expression"][["gene_name"]]
				matrix = matrix.loc[:, gene_list.index]

			   # table of gene properties
			
				if filtered:
					identity_filename = 'cell_identities.csv'
				else:
					identity_filename = 'raw_cell_identities.csv'

				print("Loading guide identities:" + os.path.join(directory, 'outs', identity_filename) + '...')
				# adjust for historical differences in column names...
				guide_identities = pd.read_csv(os.path.join(directory, 'outs', identity_filename), header=None, names=['cell_barcode', 'number_of_cells', 'guide_identity', 'guide_UMI_count']).set_index('cell_barcode')
				guide_identities["guide_name"] = ['|'.join(map(lambda i: guide_list.loc[i, "gene_name"], x.split('|'))) for x in guide_identities["guide_identity"]]
						
				# table of cell properties
				if filtered:
					cell_list = pd.merge(cell_barcodes, guide_identities, left_on='cell_barcode', right_index=True, how='left').set_index('cell_barcode')
				else:
					cell_list = pd.merge(cell_barcodes.iloc[ind], guide_identities, left_on='cell_barcode', right_index=True, how='left').set_index('cell_barcode')

				cell_list['UMI_count'] = matrix.sum(axis=1)
				cell_list["gem_group"] = [int(x.split('-')[1]) for x in cell_list.index]   
				cell_list["single_cell"] = (cell_list['number_of_cells'] == 1)

			else:
				if filtered:
					identity_filename = 'cell_identities.csv'
				else:
					identity_filename = 'raw_cell_identities.csv'
				
				print("Loading guide identities:" + os.path.join(directory, 'outs', identity_filename) + '...')
				# adjust for historical differences in column names...
				guide_identities = pd.read_csv(os.path.join(directory, 'outs', identity_filename)).rename(columns={'cell BC': 'cell_barcode',
				'read count': 'guide_read_count',
				'UMI count': 'guide_UMI_count',
				'coverage': 'guide_coverage',
				'cell_BC': 'cell_barcode',
				'read_count': 'guide_read_count',
				'UMI_count': 'guide_UMI_count'}).set_index('cell_barcode')
				cols = guide_identities.columns
				cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, six.string_types) else x)
				guide_identities.columns = cols
						
				# table of cell properties
				if filtered:
					cell_list = pd.merge(cell_barcodes, guide_identities, left_on='cell_barcode', right_index=True, how='left').set_index('cell_barcode')
				else:
					cell_list = pd.merge(cell_barcodes.iloc[ind], guide_identities, left_on='cell_barcode', right_index=True, how='left').set_index('cell_barcode')
				guide_targets = cell_list['guide_identity'].map(lambda x: str(x).split('_')[0])
				guide_targets.name = 'guide_target'
				cell_list['guide_target'] = guide_targets
				
				cell_list['single_cell'] = (cell_list['number_of_cells'] == 1) & (cell_list['good_coverage']) & (~(cell_list['guide_identity'] == '*'))
				cell_list['UMI_count'] = matrix.sum(axis=1) 
		return cls(matrix, cell_list, gene_list, source=directory)



	def where(self, cells=None, genes=None, normalized=False, gene_names=False, densify=True, return_query_str=False, dropna=False, **kwargs):
		"""Return expression matrix or normalized expression matrix sliced based on properties of cells or genes
		
		Example: pop.where(cells='perturbation == "control"', genes='mean > 0.25')
		would return the expression matrix genes of mean > 0.25 within control cells
		
		pop.where(cells='index in @interesting_cells',
				  genes='gene_name in @interesting_genes',
				  interesting_cells=interesting_cells,
				  interesting_genes=interesting_genes)
		would return an expression matrix with cell barcodes taken from the list interesting_cells and genes 
		whose names are in the list interesting_genes. Note these have to be passed as keyword arguments and
		are then injected into the namespace searched by query.
		
		Args:
		
		cells: query string for cell properties (i.e. executes pop.cells.query(cells))
		
		genes: likewise for genes
		
		normalized: return the normalized expression matrix
				
		gene_names: if True, return a table with gene names as columns instead of Ensembl ids (default: False)
		
		new_population: if True, return a new CellPopulation instance subsetted according to the query strings. Equivalent to using the subpopulation method.
		
		**kwargs: dict of variables to inject into name space accessed by query, necessary if you want to for example test membership against a predefined list
		"""
		# which matrix to use
		if (normalized):
			matrix = self.normalized_matrix
		else:
			matrix = self.matrix
			
		if densify and isinstance(matrix, pd.SparseDataFrame):
			if not normalized:
				print('(where) Densifying matrix...')
			else:
				print('(where) Densifying normalized matrix...')
			matrix = matrix.to_dense()
				
		# if no queries, just return the expression matrix
		if (genes is None) & (cells is None):
			if gene_names:
				out_matrix = matrix.copy()
				out_matrix.columns = self.gene_names(out_matrix.columns)
			else:
				return matrix
		
		# are we performing a query based on traits?
		complex_gene_indexing = isinstance(genes, str)
		complex_cell_indexing = isinstance(cells, str)
		cell_query_str = ''
		gene_query_str = ''
		
		# construct cell list
		if complex_cell_indexing: # query on metadata
			cell_index = self.cells.query(cells, global_dict=kwargs, engine='numexpr').index
			cell_query_str = '| ' + cells + ' '
		else: # list of cell barcodes or no condition
			cell_index = cells
			if cells is not None:
				cell_query_str = '| cells in list '
			else: # no condition
				cell_query_str = ''
		
		# construct gene list
		if complex_gene_indexing: # query on metadata
			gene_index = self.genes.query('(' + genes + ') and (in_matrix)', global_dict=kwargs, engine='numexpr').index
			gene_query_str = '| '+ genes + ' '
		else: # list of genes or no condition
			if genes is not None: # list
				test_gene = genes[0]
				if test_gene[0:4] != 'ENSG': # we already have a list of ensembl ids
					gene_names = True # return gene names when passed a list of gene names
					genes = self.gene_ids(genes)
					if isinstance(genes, six.string_types):
						genes = [genes,]
				gene_index = self.genes.index[self.genes.index.isin(genes) & self.genes['in_matrix']]
				gene_query_str = '| genes in list ' 
			else: # no condition
				gene_query_str = ''
				gene_index = genes
		
		# construct output, separate cases because faster
		if genes is None:
			out_matrix = matrix.loc[cell_index]
		elif cells is None:
			out_matrix = matrix[gene_index]
		else: # querying on both
			out_matrix = matrix.loc[cell_index, gene_index]  
						  
		if gene_names:
			out_matrix.columns = self.gene_names(out_matrix.columns)

		if dropna:
			out_matrix = out_matrix.replace({np.inf: np.nan, -np.inf: np.nan}).dropna(axis=1)
		
		if return_query_str:
			return out_matrix, (cell_query_str, gene_query_str)
		else:
			return out_matrix
	
	def subpopulation(self, cells=None, genes=None, normalized_matrix=None, **kwargs):
		"""Return a new CellPopulation instance based on properties of cells or genes
		
		Example: 
		
		pop = pop.subpopulation(cells='(single_cell) and \
							   (guide_identity in @perturbation_list),
								perturbation_list=perturbation_list)
		would return a new CellPopulation instance consisting of cell barcodes that are called as singletons (i.e. no
		doublets), whose guide_identity is in the list perturbation_list.
		
		Note: This function internally uses pop.where
		
		Args:
		
		cells: query string for cell properties (i.e. executes pop.cells.query(cells))
		
		genes: likewise for genes
		
		return_multiplets: return cell barcodes that are expected to contain two cells based on guide identity calling (default: False)
		
		generate_normalized_matrix: whether to generate normalized expression matrix, otherwise this will have to be separately computed

		**kwargs: dict of variables to inject into name space accessed by query, necessary if you want to for example test membership against a predefined list
		"""
		new_matrix, (cell_query_str, gene_query_str) = self.where(cells=cells,
						  genes=genes,
						  return_query_str=True,
						  **kwargs)
		
		new_pop = self.__class__(new_matrix,
								 self.cells.loc[new_matrix.index],
								 self.genes.loc[new_matrix.columns],
								 source=self.source + '  ||' + gene_query_str + cell_query_str + ' || ')
		
		if normalized_matrix == 'inherit':
			print('Inheriting from parent normalized matrix...')
			new_pop.normalized_matrix = self.normalized_matrix.loc[new_matrix.index, new_matrix.columns].copy()
		else:
			new_pop.normalized_matrix = normalized_matrix

		# if some genes have been removed for memory purposes, add the names back to the genes table
		if not self.genes['in_matrix'].all():
			missing = self.genes.query('~in_matrix').copy()
			new_pop.genes = pd.concat([missing, new_pop.genes])
			new_pop.genes = new_pop.genes.loc[self.genes.index]
			
		return new_pop
	
	def fit(self, transformer, y=None, cells=None, genes=None, normalized=False, **kwargs):
		matrix = self.where(cells=cells,
						  genes=genes,
						  normalized=normalized,
						  **kwargs)
		
		transformer.fit(matrix, y)
		
	def fit_transform(self, transformer, y=None, cells=None, genes=None, normalized=False, prefix=None, return_dataframe=True, **kwargs):
		matrix = self.where(cells=cells,
						  genes=genes,
						  normalized=normalized,
						  **kwargs)
		
		Z = transformer.fit_transform(matrix, y) 

		# check if it's one of my transformers, in which case it already returns a dataframe
		if isinstance(Z, pd.DataFrame):
			if return_dataframe:
				return Z
			else:
				return Z.values
		else:
			if return_dataframe:
				Z = pd.DataFrame(Z, index=matrix.index)
				if prefix is not None:
					Z.columns = [prefix + '{0}'.format(i) for i in Z.columns]
				return Z
			else:
				return Z
		
	def groupby(self, key_name, densify=True, **kwargs):
		"""Provides an iterator to (key, expression_data) pairs for groups of cells defined by a trait.
		Each key corresponds to one of the unique values of a chosen column in the pop.cells metadata.
		groupby then returns the expression data subsetted to cells that belong to that subpopulation,
		along with any additional conditions (e.g. on mean expression).
		
		Example:
			means = pd.DataFrame({k: v.mean() for k, v in pop.groupby('guide_identity', genes='mean > 1')})
			
		would produce a new dataframe whose columns are the mean expression profiles of all genes
		with mean > 1 within the subpopulations defined by guide identity.
		
		Args:
		
		key_name: name of the column in pop.cells that defines the categories. Each unique value will
		be used to define a subpopulation.
		
		All other parameters are passed to pop.where and can be used to further subset genes or cells.
		"""
		
		cell_str = 'index in @key_barcodes'
		if 'cells' in kwargs:
			cells = kwargs.pop('cells')
			if cells is not None:
				cell_str = cell_str + ' and ({0})'.format(cells)
			print('groupby: {0} (key = {1})'.format(cell_str, key_name))
		else:
			cells = None
				
		if 'normalized' in kwargs:
			data_normalized = kwargs.get('normalized')
		else:
			data_normalized = False
			
		# if the data matrix is sparse we densify it before iterating so that 
		# this operation is performed only once
		if not data_normalized and isinstance(self.matrix, pd.SparseDataFrame) and densify:
			sparsify = True
			self.densify_matrix()
		elif data_normalized and isinstance(self.normalized_matrix, pd.SparseDataFrame) and densify:
			sparsify = True
			self.densify_normalized_matrix()
		else:
			sparsify = False

		if cells is None:
			keys = sorted(self.cells[key_name].unique())
		else:
			data = self.where(cells=cells, densify=densify, **kwargs)
			keys = sorted(self.cells.loc[data.index, key_name].unique())
			
		# generator for submatrices
		for key in keys:
			key_barcodes = self.cells[self.cells[key_name] == key].index
			data = self.where(cells=cell_str, densify=densify, key_barcodes=key_barcodes, **kwargs)
			yield key, data

		# if we densified matrix then return it to sparse
		if sparsify and densify:
			if data_normalized:
				self.sparsify_normalized_matrix()
			else:
				self.sparsify_matrix()
			
	def groupby_values(self, key_name, **kwargs):
		for _, value in self.groupby(key_name, **kwargs):
			yield value
						
	def groupby_apply(self, key_name, function_dict, verbose=False, show_progress=False, **kwargs):
		"""Applies a function or group of functions to summarize gene expression in subpopulations of cells
		defined by a trait. E.g. Mean and standard deviation of gene expression within groups of cells
		with different perturbations. To make calculations faster, a dictionary of functions to apply
		is passed, so that operations on the same data can all be done in series.
		
		Example:
			stats = pop.groupby_apply('guide_identity', {'mean': 'mean', 'numpy_mean': np.mean})
			
		This returns a dictionary where stats['mean'] contains, for example, a DataFrame of mean expression
		profiles within each guide-bearing subpopulation. Note that string function names can be supplied.
		These will be mapped to the pandas equivalent: e.g. 'sum' will use df.sum(), and 'std' will use
		df.std(). This ability is included because these implementations are generally significantly
		faster than passing the equivalent numpy functions.
		
		Args:
		
		key_name: name of the column in pop.cells that defines the categories. Each unique value will
		be used to define a subpopulation.
		
		function_dict: dictionary consisting of (function_name, string) or (function_name, function) pairs.
		String functions will be dispatched to the equivalent pandas function as described above. Functions
		will be applied as df.apply(function).
		
		All other parameters are passed to pop.groupby, and hence to pop.where, and can thus be used to
		further subset genes or cells.        
		"""
		calcs = defaultdict(dict)
		keys = list()
		
		if show_progress:
			groups = progress(self.groupby(key_name, **kwargs))
		else:
			groups = self.groupby(key_name, **kwargs)
			
		for key, data in groups:
			if show_progress:
				groups.set_description('Applying to key {0}...'.format(key))
			if verbose:
				print('Applying to key {0}...'.format(key))
			keys.append(key)
		
			for func_name, func in six.iteritems(function_dict):
				# we are wrapping a method implemented in pandas... this is to access methods like
				# df.sum(), df.std(), etc., which are implemented in cython and so run much faster
				# than using apply with corresponding numpy functions
				if isinstance(func, six.string_types):
					calcs[key][func_name] = getattr(data, func)()
				else:
					calcs[key][func_name] = data.apply(func)
					
		# now swap inner and outer keys of dictionaries and convert to DataFrame
		out = defaultdict(dict)
		for func_name in function_dict.keys():
			for key in keys:
				out[func_name][key] = calcs[key][func_name]
			out[func_name] = pd.DataFrame(out[func_name]).T
			
		if len(function_dict) == 1:
			return out[list(function_dict.keys())[0]]
		else:
			return out

	def groupby_map(self, key_name, function_dict, verbose=False, show_progress=False, **kwargs):
		"""General mapping of a function or group of functions to subpopulations of cells
		defined by a trait. E.g. Mean UMI count of cells bearing a given perturbation, or the number of cells
		bearing a perturbation. To make calculations faster, a dictionary of functions is passed, so that 
		operations on the same data can all be done in series. For scalar-valued functions the output is 
		returned as a Series or DataFrame according to the number of functions passed.
		
		Example:
			pop.groupby_map('guide_identity',
							{'num_cells': lambda x: x.shape[0], 'UMI_mean': lambda x: x.mean().sum()})
							
		will return a DataFrame with the number of cells and the mean UMI count in each guide-bearing
		subpopulation. 
		
		General-valued functions can also be used:
		
			pop.groupby_map('guide_identity',
							{'percent': lambda x: x.div(x.sum(axis=1), axis=0)})
							
		will return a dictionary of DataFrames indexed by guide_identity where the expression matrices
		have been normalized such that total gene expression in each cell sums to 1.
				
		Args:
		
		key_name: name of the column in pop.cells that defines the categories. Each unique value will
		be used to define a subpopulation.
		
		function_dict: dictionary consisting of (function_name, function) pairs. Functions should return
		a single value for each subpopulation.
		
		All other parameters are passed to pop.groupby, and hence to pop.where, and can thus be used to
		further subset genes or cells.        

		"""
		is_scalar = True
		calcs = defaultdict(dict)
		
		if show_progress:
			groups = progress(self.groupby(key_name, **kwargs))
		else:
			groups = self.groupby(key_name, **kwargs)
			
		for key, data in groups:
			if show_progress:
				groups.set_description('Applying to key {0}...'.format(key))
			if verbose:
				print('Applying to key {0}...'.format(key))

			for func_name, func in six.iteritems(function_dict):
				func_value = func(data)
				if hasattr(func_value, '__len__') and (not isinstance(func_value, str)):
					# if non-scalar function(s), we won't do formatting to output
					is_scalar = False
				calcs[key][func_name] = func_value
						   
		# now swap inner and outer keys of dictionaries and convert to DataFrame
		out = defaultdict(dict)
		for func_name in function_dict.keys():
			for key in calcs.keys():
				out[func_name][key] = calcs[key][func_name]
				
		if is_scalar:
			if len(function_dict) == 1:
				func_name = list(function_dict.keys())[0]
				return pd.Series(out[func_name], name=func_name)
			else:
				return pd.DataFrame(out)
		else: # for non-scalar functions just return dictionary of values
			if len(function_dict) == 1:
				return out[list(function_dict.keys())[0]]
			else:
				return out
		
	def average(self, key_name, verbose=False, **kwargs):
		print('Computing average expression matrices...')
		matrix = self.groupby_apply(key_name, {'mean': 'mean'},
								   verbose=verbose, **kwargs)
		if self.normalized_matrix is not None:
			print('Computing normalized average expression matrices...')
			normalized_matrix = self.groupby_apply(key_name, {'mean': 'mean'},
												  normalized=True,
												  verbose=verbose, **kwargs)
		else:
			normalized_matrix = None
		print('Computing clusters...')
		cell_list = self.groupby_map(key_name,
									{'num_cells': lambda x: x.shape[0],
									 'UMI_count': lambda x: x.mean().sum()},                                    
									verbose=verbose, **kwargs)
		gene_list = self.genes.loc[matrix.columns][['gene_name']].copy()
	
		mean_pop = MeanPopulation(matrix, cell_list, gene_list, normalized_matrix=normalized_matrix)
			   
		# if some genes have been removed for memory purposes, add the names back to the genes table
		if not self.genes['in_matrix'].all():
			missing = self.genes.query('~in_matrix').copy()
			mean_pop.genes = pd.concat([missing, mean_pop.genes])
			mean_pop.genes = mean_pop.genes.loc[self.genes.index]

		return mean_pop
				
	def metaapply(self, function_dict, axis=1, normalized=False, **kwargs):
		if not normalized:
			df = self.matrix
		else:
			df = self.normalized_matrix
		
		if axis == 1:
			meta_df = self.cells
		else:
			meta_df = self.genes
			
		return metaapply(meta_df, df, function_dict, **kwargs)
	
	def gene_ids(self, gene_list, drop_duplicates='first'):
		if isinstance(gene_list, six.string_types):
			gene_list = [gene_list]
		test_gene = gene_list[0]
		if test_gene[0:4] == 'ENSG': # we already have a list of ensembl ids
			return gene_list
		
		deduplicated_list = self.genes.query('gene_name in @gene_list') \
									  .sort_values('mean', ascending=False) \
									  .drop_duplicates(subset='gene_name', keep=drop_duplicates) \
									  .reset_index()
					
		if deduplicated_list.empty:
			return ''

		# this sorts the ids that have been found into the query order
		deduplicated_list['gene_name'] = deduplicated_list['gene_name'].astype("category")
		deduplicated_list['gene_name'].cat.set_categories(gene_list, inplace=True)
		deduplicated_list.sort_values('gene_name', inplace=True)
		
		if len(deduplicated_list) == 1:
			return deduplicated_list['gene_id'].values[0]
		else:
			return deduplicated_list['gene_id'].values
			
	def gene_names(self, gene_list):
		# list of genes
		if hasattr(gene_list, '__len__') and (not isinstance(gene_list, str)):
			test_gene = gene_list[0]
			if test_gene[0:4] == 'ENSG':
				return self.genes.loc[gene_list, 'gene_name'].values
			else:
				return gene_list
		# single string
		else:
			return self.genes.loc[gene_list, 'gene_name']
	
	def add_property(self, cells=None, genes=None, name=None):
		""" Add a property describing either cells or genes to a CellPopulation.
		
		Example: pop.add_property(cells=perturbation_targets)

		Args:
		
		cells: Add the dataframe or series to the cells table of the CellPopulation. Any existing columns in the cells table with the same name will be replaced by the new columns.
		
		genes: Same, but add to genes table instead.
		
		name: If adding a pandas Series, the column name that should be used. For dataframes the names are taken from the columns.
		"""
		if cells is not None:
			if isinstance(cells, pd.Series):
				if name is not None:
					cells.name = name
				cells = pd.DataFrame(cells)
			else:
				if name is not None:
					cells.columns = name
			# replace any pre-existing columns with the same name
			cols_to_use = self.cells.columns.difference(cells.columns)
			self.cells = pd.merge(self.cells[cols_to_use], cells, left_index=True, right_index=True, how='left')
			
		if genes is not None:
			if isinstance(genes, pd.Series):
				genes=pd.DataFrame(genes)
				if name is not None:
					genes.columns = name
			else:
				if name is not None:
					genes.columns = name
			# replace any pre-existing columns with the same name
			cols_to_use = self.genes.columns.difference(genes.columns)
			self.genes = pd.merge(self.genes[cols_to_use], genes, left_index=True, right_index=True, how='left')         
			
	def sparsify_matrix(self, fill_value=0):
		print('Sparsifying matrix...')
		self.matrix = self.matrix.to_sparse(fill_value=fill_value)
		
	def densify_matrix(self, fill_value=0):
		print('Densifying matrix...')
		self.matrix = self.matrix.to_dense()
   
	def sparsify_normalized_matrix(self, fill_value=0):
		print('Sparsifying normalized matrix...')
		self.normalized_matrix = self.normalized_matrix.to_sparse(fill_value=fill_value)

	def densify_normalized_matrix(self, fill_value=0):
		print('Densifying normalized matrix...')
		self.normalized_matrix = self.normalized_matrix.to_dense()
			
	def plot(self, data, clusters=None, traits=None, gene=None, normalized=True, cluster_subset=None, cluster_size_threshold=0, cm=None, randomize_cmap=False, s=10, alpha=1, auxiliary_matrix=None, no_axes=True, no_cbar=False):
		"""Plot a CellPopulation
		
		Args:
		
		data: Dataframe of coordinates to plot. It is assumed that the first two columns are the X and Y coordinates.
		
		clusters: Color based on discrete categories in column from cell table (i.e. pop.cells[clusters])
		
		traits: Color by continuous variable in column from cell table (i.e. pop.cells[traits])
		
		gene: Plot expression of a gene with the given name or gene id
		
		normalized: whether to plot normalized expression of gene (default: True)
		
		cluster_subset: plot only the values of clusters within this list
		
		cluster_size_threshold: do not plot discrete clusters smaller than this size
		
		cm: override default colormaps used for plotting
		
		randomize_cmap: randomize order of colormap (useful for discrete data sets with similar colors that end up next to each other by chance)
		
		s: size of marker (default: 10)
		
		alpha: alpha value for plotting (default: 1)
		
		auxiliary_matrix: pull expression data from an alternative source (e.g. imputed data)
		
		no_axes: remove axes from plot (default: True, as for t-sne plots the axes are meaningless) 
		
		no_cbar: don't plot color bar (default: False)
		"""
		

		X = data[data.columns[0]]
		Y = data[data.columns[1]]

		if clusters is not None:
			if isinstance(clusters, six.string_types): # using a property name
				clusters = self.cells.loc[data.index][clusters]
			
			if cluster_subset is None:
				unique_clusters = np.unique(clusters)
			else:
				unique_clusters = cluster_subset

			cluster_sizes = pd.Series({cluster: np.sum(clusters==cluster) for cluster in unique_clusters})
			clusters_to_plot = cluster_sizes[cluster_sizes > cluster_size_threshold].index.values
			num_clusters = len(clusters_to_plot)
			
			if cm is None:
				cm = sns.hls_palette(num_clusters)
				if randomize_cmap:
					rand_order = np.random.permutation(num_clusters)
					cm = np.array(cm)
					cm = cm[rand_order]
			
			for i, cluster in enumerate(clusters_to_plot):
				 if np.sum(clusters == cluster) > cluster_size_threshold:       
					plt.scatter(X[clusters == cluster], Y[clusters == cluster], color=cm[i], s=s, alpha=alpha)

			cax = plt.gca()
			box = cax.get_position()
			cax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
			cax.legend(labels=clusters_to_plot, loc='center left', bbox_to_anchor=(1, 0.5))
		   
			#plt.legend(labels=clusters_to_plot, fancybox=True)
		elif traits is not None:
			if isinstance(traits, six.string_types): # using a property name
				traits = self.cells.loc[data.index][traits]
			
			if cm is None:
				cm = 'bwr'
			cax = plt.scatter(X, Y, c=traits.values, cmap=cm, s=s)
			fig = plt.gcf()
			if not no_cbar:
				cbar = fig.colorbar(cax)
			#plt.clim(np.percentile(traits, [5, 95]))            
			
		elif gene is not None:
			gene = self.gene_ids(gene)
			if auxiliary_matrix is not None:
				data = auxiliary_matrix.loc[data.index][gene]
			elif normalized:
				data = self.normalized_matrix.loc[data.index][gene]
			else:
				data = self.matrix.loc[data.index][gene]
			
			if cm is None:
				cm = 'bwr'
			cax = plt.scatter(X, Y, c=data.values, cmap=cm, s=s)
			fig = plt.gcf()
			if not no_cbar:
				cbar = fig.colorbar(cax)
			
		if no_axes:
			ax = plt.gca()
			ax.set_axis_off()
	
	def plots(self, data, properties, types=None, randomize_cmap=False, normalized=True, auxiliary_matrix=None, s=10, alpha=1, no_axes=True, tight_layout=True):
		"""Plot multiple properties of a CellPopulation
		
		Args:
		
		data: Dataframe of coordinates to plot. It is assumed that the first two columns are the X and Y coordinates.

		properties: list of column names from pop.cells
		
		types: list of plot types. If None, all are assumed to be continuous and are plotted as heatmaps, if a single word ('discrete', 'trait' or 'gene'), all are plotted as that type, if an array, corresponding entry from properties will be plotted as that data type
		
		normalized: whether to plot normalized expression of gene (default: True)
		
		randomize_cmap: randomize order of colormap (useful for discrete data sets with similar colors that end up next to each other by chance)
		
		s: size of marker (default: 10)
		
		alpha: alpha value for plotting (default: 1)
		
		auxiliary_matrix: pull expression data from an alternative source (e.g. imputed data)
		"""
		num_graphs = len(properties)
	
		if types is None or types is 'discrete':
			types = ('trait',)*num_graphs
		elif len(types) == 1:
			types = (types,)*num_graphs

		for i, prop in enumerate(properties):
			plt.subplot(1, num_graphs, i + 1)
			if (types[i] == None) or (types[i] == 'discrete'):
				self.plot(data, prop, randomize_cmap=randomize_cmap, s=s, alpha=alpha, no_axes=no_axes)
				plt.xlabel(prop)
			elif types[i] == "trait":
				self.plot(data, traits=prop, s=s, alpha=alpha, no_axes=no_axes)
				plt.xlabel(prop)
			elif types[i] == "gene":
				self.plot(data, gene=prop, normalized=normalized, auxiliary_matrix=auxiliary_matrix, s=s, alpha=alpha, no_axes=no_axes) 
				plt.xlabel(prop)
				
		if tight_layout:
			plt.tight_layout()
			
	def info(self):
		print('Matrix')
		print('==========================')
		self.matrix.info(memory_usage='deep')
		if self.normalized_matrix is not None:
			print('\nNormalized matrix')
			print('==========================')
			self.normalized_matrix.info(memory_usage='deep')
			
class MeanPopulation(CellPopulation):
	
	def __init__(self, matrix, cell_list, gene_list, source='arrays', normalized_matrix=None, calculate_statistics=True):
		"""A class for holding mean population data with separate tables for expression, normalized expression, and data about cells and genes
		
		Args:
		
		matrix: expression matrix of UMIs per cells (rows = cell barcodes, columns = Ensembl gene ids)
		
		cell_list: table of properties of cells in the population (indexed by cell barcode)
		
		gene_list: table of properties of genes in the population (indexed by Ensembl gene id)
		
		source: keeps track of how this population was derived. If it is from raw data, the source will be 'arrays'. If it is a subpopulation derived from another population, this will contain the list of criteria that were used to create that subpopulation
				
		normalized_matrix: Add the supplied normalized expression matrix
		"""
		self.matrix = matrix
		self.normalized_matrix = normalized_matrix
		
		# fill out the list of gene properties
		if calculate_statistics:
			print("Generating summary statistics...")
			gene_list['mean'] = matrix.mean()
			gene_list['std'] = matrix.std()
			gene_list['cv'] = gene_list['std']/gene_list['mean']
			gene_list['in_matrix'] = True
			gene_list['gini'] = matrix.apply(lambda x: gini(x.values))
		
		self.genes = gene_list
		self.cells = cell_list
		
		self.source = source
				
		print("Done.")  
				
	def expression_table(self, genes, cluster_size_thresh=0, sort=True, linkage=None, optimal_ordering=False, **kwargs):
		"""Generates a table of summary statistics for a given set of genes within the subpopulations defined by the MeanPopulation

		Example: data = mean_pop.expression_table('index in @interesting_genes', interesting_genes=interesting_genes)

		Args:

		gene_list: list of gene names or ids

		cluster_size_thresh: if non-zero, drop clusters of absolute size less than this number

		sort: sort the gene list by hierarchical clustering so that related genes appear next to each other in the table (default: True). Note: In certain pathological cases (e.g. NaN expression values) the routine may fail, in which case stopping the sort will help.

		return_linkage: return the linkage for the clustering as a second argument
		"""
		unnorm_data = self.where(genes=genes, **kwargs).T
		unnorm_data.columns = pd.MultiIndex.from_tuples(zip(('mean',)*len(unnorm_data.columns), unnorm_data.columns))

		if self.normalized_matrix is not None:
			norm_data = self.where(genes=genes, normalized=True, **kwargs).T        
			norm_data.columns = pd.MultiIndex.from_tuples(zip(('weight',)*len(norm_data.columns), norm_data.columns))
			table = pd.concat([norm_data, unnorm_data], axis=1)
		else:
			table = unnorm_data

		table.insert(0, 'gene_name', self.genes.loc[table.index, 'gene_name'])
		table.insert(1, 'total_mean', self.genes.loc[table.index, 'mean'])

		# drop small clusters
		if cluster_size_thresh > 0:
			cluster_sizes = self.cells['num_cells']
			clusters = cluster_sizes[cluster_sizes >= cluster_size_thresh].index
			table = table.loc[:, table.columns.isin(clusters, level=1)]

		# sort the gene list by expression pattern across the clusters
		if linkage is not None:
			order = table.index[leaves_list(linkage)]
			table = table.loc[order]
		elif sort:
			if self.normalized_matrix is not None:
				Z = self.cluster_genes(genes, method='ward', metric='euclidean', optimal_ordering=optimal_ordering)
			else:
				Z = self.cluster_genes(genes, method='ward', metric='euclidean', normalized=False, optimal_ordering=optimal_ordering)

			order = table.index[leaves_list(Z)]
			table = table.loc[order]

		return table
	
	def cluster_genes(self, genes, method='ward', metric='euclidean', optimal_ordering=True, normalized=True, **kwargs):
		matrix = self.where(genes=genes, normalized=normalized, **kwargs)
		return linkage(matrix.T, method=method, metric=metric, optimal_ordering=optimal_ordering)
	
	def expression(self, genes, cluster_size_thresh=0, cluster_order=None, sort=True, linkage=None, optimal_ordering=False, **kwargs):
		"""Return a decorated expression table colored by expression value across clusters and with gene names linked to GeneCards

		Example: pretty_gene_table(interesting_genes)

		Args:

		gene_list: list of gene names or ids

		cluster_size_thresh: if non-zero, drop clusters of absolute size less than this number

		sort: sort the gene list by hierarchical clustering so that related genes appear next to each other in the table (default: True). Note: In certain pathological cases (e.g. NaN expression values) the routine may fail, in which case stopping the sort will help.

		cluster_order: put the cluster columns in the order specified by this list
		"""
		# get cluster statistics
		data = self.expression_table(genes, cluster_size_thresh=cluster_size_thresh, sort=sort, linkage=linkage, optimal_ordering=optimal_ordering)
		n_clusters = len(data['weight'].columns)

		# GeneCard links for gene names
		data['gene_name'] = data['gene_name'].apply(lambda x: '<a href="http://www.genecards.org/cgi-bin/carddisp.pl?gene={0}">{1}</a>'.format(x, x))

		# reorder columns if desired
		if cluster_order is not None:
			order = [('gene_name', '')]
			order.append(('total_mean', ''))
			order.extend(zip(('weight',)*len(cluster_order), cluster_order))
			order.extend(zip(('mean',)*len(cluster_order), cluster_order))
			data = data[order]

		# color expression table
		cm = sns.diverging_palette(240, 10, n=9, as_cmap=True)
		s = data.style.background_gradient(cmap=cm, axis=1, low=0.5, high=0.5, subset=data.columns[2:2 + n_clusters]) \
							 .background_gradient(cmap=cm, axis=1, low=0.5, high=0.5, subset=data.columns[n_clusters + 2:]) \
							 .format("{:.2f}", subset=data.columns[2:])
		return s
	
	def expression_heatmap(self, genes, cluster_size_thresh=0, cluster_order=None, linkage=None, normalized=False, **kwargs):
		"""Plot a heatmap of gene expression across clusters

		gene_list: list of gene names or ids

		cluster_size_thresh: if non-zero, drop clusters of absolute size less than this number

		cluster_order: put the cluster columns in the order specified by this list     

		linkage: if not None, use this linkage to cluster genes

		square: plot clustermap using square aspect ratio (default: True)

		**kwargs: all additional keyword arguments are passed on to Seaborn clustermap
		"""
		# get cluster statistics
		data = self.expression_table(genes, cluster_size_thresh=cluster_size_thresh, cluster_order=cluster_order, sort=False)
		if normalized:
			data = data.set_index('gene_name')['weight']
		else:
			data = data.set_index('gene_name')['mean']

		# reorder columns if desired
		if cluster_order is not None:
			data = data[cluster_order]

		if linkage is not None:
			cg = sns.clustermap(data.T,
				   col_linkage=linkage,
				   row_cluster=False,
				   **kwargs)
		else:
			cg = sns.clustermap(data.T,
							   row_cluster=False,
							   **kwargs)

		plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0);
		return cg

def gini(array):
	"""Calculate the Gini coefficient of a numpy array."""
	# From: https://github.com/oliviaguest/gini
	# based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
	# from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
	array = array.flatten() #all values are treated equally, arrays must be 1d
	if np.amin(array) < 0:
		array -= np.amin(array) #values cannot be negative
	array += 0.0000001 #values cannot be 0
	array = np.sort(array) #values must be sorted
	index = np.arange(1,array.shape[0]+1) #index per array element
	n = array.shape[0]#number of array elements
	return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient
	
def correlation_heatmap(pop, gene_list, corr_method='pearson', normalized=True, linkage=None, robust=True, method=None, metric=None, **kwargs):
	"""Heatmap of correlations of gene expression

	Example: pop.correlation_heatmap(interesting_genes)

	Returns the Seaborn ClusterGrid instance

	Args:

	gene_list: list of genes to compute correlation matrix for

	corr_method: type of correlation coefficient (default: 'pearson')

	normalized: if True, use normalized expression values (default: True)

	linkage: if supplied, use this linkage to order the heatmap

	robust: if True, use robust colormap range

	method: method to use for hierarchical clustering

	metric: metric to use for hierarchical clustering

	**kwargs: all additional arguments are passed to Seaborn's clustermap        
	"""
	if normalized:
		corr_matrix = pop.normalized_matrix[gene_list].corr(method=corr_method)
	else:
		corr_matrix = pop.matrix[gene_list].corr(method=method)

	corr_matrix.columns = pop.genes.loc[corr_matrix.columns, 'gene_name']
	corr_matrix.index = pd.Index(pop.genes.loc[corr_matrix.index.values, 'gene_name'])

	cmap = sns.diverging_palette(220, 10, as_cmap=True)

	if linkage is not None:
		cg = sns.clustermap(corr_matrix,
							cmap=cmap,
							square=True,
							row_linkage=linkage,
							col_linkage=linkage,
							robust=robust,
							**kwargs)
	elif (metric is not None) or (method is not None):
		cg = sns.clustermap(corr_matrix,
							cmap=cmap,
							square=True,
							metric=metric,
							method=method,
							robust=robust,
							**kwargs)
	else:
		cg = sns.clustermap(corr_matrix,
							cmap=cmap,
							square=True,
							method='ward',
							robust=robust,
							**kwargs)


	plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0);

	return cg

def fancy_dendrogram(*args, **kwargs):
	"""Plot an annotated dendrogram
	"""
	max_d = kwargs.pop('max_d', None)
	if max_d and 'color_threshold' not in kwargs:
		kwargs['color_threshold'] = max_d
	annotate_above = kwargs.pop('annotate_above', 0)

	ddata = dendrogram(*args, **kwargs)

	if not kwargs.get('no_plot', False):
		for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
			x = 0.5 * sum(i[1:3])
			y = d[1]
			if y > annotate_above:
				plt.plot(x, y, 'o', c=c)
				plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
							 textcoords='offset points',
							 va='top', ha='center')
		if max_d:
			plt.axhline(y=max_d, c='k')
	return ddata

def fit_dendrogram(population, data, max_d, method='ward', metric='euclidean', on='genes', return_linkage=False):   
	"""Perform hierachical clustering and plot a distance threshold used to define clusters
	
	Example: 
	
	clusters, order = fit_dendrogram(pop, data, 2, on='genes')
	
	Returns a list of clusters and the order they appear in the dendrogram. (order is a list such that clusters[order] is the list of genes in the same order as in the dendrogram.)
	
	Args:
	
	population: The population from which gene expression data is pulled
	
	data: The matrix used for clustering (e.g. expression, a distance matrix, etc.)
	
	max_d: The maximum cophenetic distance used to define clusters (i.e., height at which dendrogram is cut)
	
	method: method used for clustering (default: 'ward')
	
	metric: distance metric used for clustering (default: 'euclidean')
	
	on: which axis to pull labels from, depending on whether you are clustering genes or cells (default: 'genes')
	
	return_linkage: If True, return the linkage as a third argument (default: False)
	"""
	# calculate linkage
	Z = linkage(data, method=method, metric=metric)

	# form clusters by max distance
	clusters = fcluster(Z, max_d, criterion='distance')
	
	# label function for dendrogram leaves
	if on == 'genes':
		labeler=lambda id: population.genes.loc[data.index[id], 'gene_name'] + ' ({0})'.format(clusters[id])
		clusters = pd.Series(clusters, index=data.columns)
	elif on == 'cells':
		labeler=lambda id: '{0} ({1})'.format(data.index[id], clusters[id])
		clusters = pd.Series(clusters, index=data.index)

	# draw the dendrogram
	ddata = fancy_dendrogram(Z,
					   leaf_font_size=10,
					   leaf_label_func=labeler,
					   max_d=max_d,
					   get_leaves=True)

	if return_linkage:
		return clusters, ddata['leaves'], Z
	else:
		return clusters, ddata['leaves']

def metaapply(meta_df, df, function_dict, show_progress=False, **kwargs):
	"""Iterates through the rows of a metadata DataFrame and applies a function(s)
	with acccess to a second DataFrame. E.g. loop through cell metadata with access
	to expression. Any missing keys are replaced with nan values.
	
	Example: metaapply(meta_data, expr_data,
					   {'target_expr': lambda meta, expr: expr.loc[meta.name, meta['guide_target']],
					   })
	returns a Series indexed by the meta_data DataFrame's index that contains, e.g.,
	the expression of the target of the guide in that cell/population. Multiple functions
	can be applied via the dictionary,
	"""
	calcs = dict()
	missing_key = False
	
	if show_progress:
		rows = progress(meta_df.iterrows())
	else:
		rows = meta_df.iterrows()
	
	for (key, meta_row) in rows:
		calcs[key] = dict()
		
		for func_name, func in six.iteritems(function_dict):
			try:
				calcs[key][func_name] = func(meta_row, df, **kwargs)
			except KeyError:
				calcs[key][func_name] = np.nan
				missing_key = True
	
	if missing_key:
		warnings.warn('Function application containing missing keys that were replaced by nans.')
			
	# now swap inner and outer keys of dictionaries and convert to DataFrame
	out = defaultdict(dict)
	for func_name in function_dict.keys():
		for key in calcs.keys():
			out[func_name][key] = calcs[key][func_name]

	if len(function_dict) == 1:
		func_name = list(function_dict.keys())[0]
		return pd.Series(out[func_name], name=func_name)
	else:
		return pd.DataFrame(out)