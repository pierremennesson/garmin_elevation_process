import osmnx as ox
import networkx as nx
import time
from multiprocessing import Pool,cpu_count
import glob
import numpy as np
import pickle
import warnings
from lib_garmin_elevation import *

place_name='Corse, France'
file_paths=glob.glob('./garmin_activities/*.gpx')
navigation_df_path='./navigation_df.csv'
save_navigation_data=False

max_line_length=float('inf')
max_projection_distance=20,
max_id_segment_gap=2
max_time_gap=100.
max_distance_gap=250.
max_delta_T=float('inf')
overlap_coeff=0.
harmonizing_step=5.
discarding_threshold=30.
overlay_threshold=0.25
affine_threshold=2.5
correlation_treshold=0.9
min_tree_components=1
min_count=2
min_cover_length=2000.
intermediate_distance=1500
min_samples_leaf=25
min_impurity_decrease=0.66*float('1e-5')

nodes_elevation_data_path='./nodes_elevation_data.p'
edges_elevation_data_path='./edges_elevation_data.p'





if __name__ == "__main__":
	t1=time.time()
	G_osm_dir=ox.graph_from_place(place_name,network_type='drive')
	t2=time.time()
	print('loading osm graph took %f s'%(t2-t1))

	G_osm=to_multi_graph(G_osm_dir)
	crs=ox.graph_to_gdfs(G_osm,edges=False).estimate_utm_crs()
	G_osm=ox.project_graph(G_osm,to_crs=crs)
	add_missing_geometries(G_osm,max_line_length=max_line_length)
	t3=time.time()
	print('processing osm graph took %f s'%(t3-t2))

	def parallelized_preprocess(file_paths):
	    garmin_dfs=[preprocess(file_path,G_osm,crs=crs,verbose=False) for file_path in file_paths]
	    return pd.concat(garmin_dfs)

	nb_cpu=cpu_count()-1
	chunked_file_paths=chunk(file_paths,nb_cpu)
	with Pool(nb_cpu) as p:
		navigation_dfs=p.map(parallelized_preprocess,chunked_file_paths)
	navigation_df=pd.concat(navigation_dfs)
	if save_navigation_data:
		navigation_df=navigation_df.to_crs('epsg:4326')
		navigation_df.to_csv(navigation_df_path,index=False)
		navigation_df=navigation_df.to_crs(crs)
	G_navigation=build_multidigraph(G_osm,navigation_df)


	t4=time.time()
	print('loading and processing navgiation data took %f s'%(t4-t3))

	dual_G=build_dual_graph(G_navigation,max_id_segment_gap=max_id_segment_gap)
	dual_tree=build_dual_tree(dual_G)
	paths=graph_decomposition(dual_tree)
	t5=time.time()
	print('computing longest paths took %f s'%(t5-t4))

	warnings.filterwarnings('ignore',category=RuntimeWarning)




	all_nodes_data,all_edges_data={},{}
	for path in paths:
		output=collect_elevation_data_from_path(G_navigation,G_osm,path,
	                            max_id_segment_gap=max_id_segment_gap,max_time_gap=max_time_gap,max_distance_gap=max_distance_gap,
	                            max_delta_T=max_delta_T,overlap_coeff=overlap_coeff,harmonizing_step=harmonizing_step,
	                            discarding_threshold=discarding_threshold,overlay_threshold=overlay_threshold,affine_threshold=affine_threshold,
	                            correlation_treshold=correlation_treshold,min_tree_components=min_tree_components,min_count=min_count,
	                            min_cover_length=min_cover_length,intermediate_distance=intermediate_distance,
	                            min_samples_leaf=min_samples_leaf,min_impurity_decrease=min_impurity_decrease)
		if output is not None:
		    nodes_data,edges_data=output
		    for node,data in nodes_data.items():
		        if not(node in all_nodes_data):
		            all_nodes_data[node]=[]
		        all_nodes_data[node]+=data
		    for edge,data in edges_data.items():
		        if not(edge in all_edges_data):
		            all_edges_data[edge]=[]
		        all_edges_data[edge]+=data

	t6=time.time()
	print('extracting elevation data from %i paths took %f s'%(len(paths),t6-t5))

	print('%i nodes elevation retrieved'%len(all_nodes_data))

	retrieved_edges=list(all_edges_data.keys())
	reversed_retrieved_edges=[(edge[1],edge[0],edge[2]) for edge in retrieved_edges]
	redundant_edges_count=len(set(retrieved_edges).intersection(reversed_retrieved_edges))//2
	print('%i edges elevation retrieved and %i edges appeared twice '%(len(retrieved_edges)-redundant_edges_count,redundant_edges_count))
	
	length_covered=np.sum([elem['length'] for L in all_edges_data.values() for elem in L])
	print('%f kms covered '%(length_covered/1000))

	mean_cover=np.mean([elem['cover'] for L in all_edges_data.values() for elem in L])
	print('mean edge cover : %f'%mean_cover)
	
	stds=[np.std(elevations) for elevations in all_nodes_data.values() if len(elevations)>1]
	if len(stds)>0:
		print('mean node elevation deviation : %f'%np.mean(stds))

	with open(nodes_elevation_data_path,'wb') as file:
		pickle.dump(all_nodes_data,file)

	with open(edges_elevation_data_path,'wb') as file:
		pickle.dump(all_edges_data,file)

	t7=time.time()
	print('saving data took %f s'%(t7-t6))
	print('total : %f s'%(t7-t1))