import osmnx as ox
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely.ops import nearest_points
import gpxpy
from itertools import groupby,combinations,product
from shapely.geometry import Point,LineString
import time
from sklearn.tree import DecisionTreeRegressor,_tree


#OSM GRAPH CORRECTION

def to_multi_graph(G_osm):
    """This function turn the osm multidigraph to a multigraph.
    One cannot just call nx.MultiGraph() because the edge data depends on the orientation.

    Parameters
    ----------
    G_osm : an osm multidigraph


    Returns
    -------
    the induced multigraph

    """
    new_edges=[]
    for u,v,k in G_osm.edges(keys=True):
        if not( (v,u,k) in new_edges):
            new_edges.append((u,v,k))
    return nx.edge_subgraph(G_osm,new_edges)




def add_missing_geometries(G_osm,max_line_length=float('inf')):
    """This function adds a geometry to the osmn edges when it's missing
    by adding a straight line.


    Parameters
    ----------
    G_osm : an osm multigraph

    max_line_length : the maximum distance between two points for a line to be added


    """
    attr,to_be_removed={},[]
    for u,v,k,d in G_osm.edges(data=True,keys=True):
        removed=False
        if not('geometry' in d):
            if 'length' in d:
                if d['length']<max_line_length:
                    ls=LineString([[G_osm.nodes()[u]['x'],G_osm.nodes()[u]['y']],[G_osm.nodes()[v]['x'],G_osm.nodes()[v]['y']]])
                    d.update({'geometry':ls})
                    attr[(u,v,k)]=d
                else:
                    removed=True
            else:
                removed=True
        if removed:
            to_be_removed.append((u,v,k))
    G_osm.remove_edges_from(to_be_removed)
    nx.set_edge_attributes(G_osm,attr)


# #GPX DATA COLLECTION
def chunk(file_paths,nb_cpu):
    """This function splits the list of file_paths 
    to be processed in list of lists to be multiprocessed

    Parameters
    ----------
    file_paths : a list of gpx files paths

    nb_cpu : the number of list of lists, should
    be equal to the number of avalaible cpus


    Returns
    -------
    list of lists of files paths



    """
    N=len(file_paths)
    n=max(N//nb_cpu,1)
    list_of_file_paths=[file_paths[i:i+n] for i in range(0,N,n)]
    if len(list_of_file_paths)>nb_cpu:
        last=list_of_file_paths[-1]
        list_of_file_paths=list_of_file_paths[:-1]
        for k,elem in enumerate(last):
            list_of_file_paths[k].append(elem)
    return list_of_file_paths


def get_points_from_activity(file_path):
    """This function reads a gpx file and returns
    the geodataframe of the gpx points
     

    Parameters
    ----------
    file_path : the path to the gpx file


    Returns
    -------
    a geodataframe with the latitude/longitude/elevation/timestamps of the gpx points

    """
    gpx_file = open(file_path, 'r')
    gpx = gpxpy.parse(gpx_file)
    try:
        data=gpx.tracks[0].segments[0].points
    except:
        data=gpx.waypoints
    return gpd.GeoDataFrame([{'file_path':file_path,'geometry':Point(pt.longitude,pt.latitude),
        'elevation':pt.elevation,'time':pt.time} for pt in data],geometry='geometry',crs="EPSG:4326")






def add_segments(navigation_df):
    """This function assigns a pre_segment/segment/orientation 
    to each points of the dataframe (see chapter (I) of the documentation)
     

    Parameters
    ----------
    navigation_df : a geodataframe containing the navigation points


    Returns
    -------
    the updated geodataframe 

    """
    navigation_df['pre_segment']=None 
    navigation_df['orientation']=None 
    navigation_df['segment']=None 

    edge_sequences_lengths=[len(list(g)) for _,g in groupby(list(navigation_df['edge']))]
    pre_segment_indexes=[sum(edge_sequences_lengths[:i]) for i in range(len(edge_sequences_lengths)+1)]
    pre_segments=[i  for k in range(len(pre_segment_indexes)-1) for i in [k]*(pre_segment_indexes[k+1]-pre_segment_indexes[k])]
    navigation_df.loc[:,'pre_segment']=pre_segments
    navigation_df=navigation_df.drop_duplicates(subset=['pre_segment','edge_coordinate'])

    edge_sequences_lengths=[len(list(g)) for _,g in groupby(list(navigation_df['edge']))]
    pre_segment_indexes=[sum(edge_sequences_lengths[:i]) for i in range(len(edge_sequences_lengths)+1)]
    pre_segments=[i  for k in range(len(pre_segment_indexes)-1) for i in [k]*(pre_segment_indexes[k+1]-pre_segment_indexes[k])]
    navigation_df.loc[:,'pre_segment']=pre_segments

    for k in range(len(pre_segment_indexes)-1):
        edge_df=navigation_df.iloc[pre_segment_indexes[k]:pre_segment_indexes[k+1]]
        if len(edge_df)==1:
            edge_df.loc[:,'orientation']=0.
        else:
            orientation=np.sign(np.diff(edge_df['edge_coordinate']))
            edge_df.iloc[1:].loc[:,'orientation']=orientation
            edge_df.iloc[0:1].loc[:,'orientation']=orientation[0]
            
    oriented_edge_sequences_lengths=[len(list(g)) for _,g in groupby(list(zip(navigation_df['pre_segment'],navigation_df['orientation'])))]
    segment_indexes=[sum(oriented_edge_sequences_lengths[:i]) for i in range(len(oriented_edge_sequences_lengths)+1)]
    segments=[i  for k in range(len(segment_indexes)-1) for i in [k]*(segment_indexes[k+1]-segment_indexes[k])]
    navigation_df.loc[:,'segment']=segments

    return navigation_df


def preprocess(file_path,G_osm,max_projection_distance=20,crs=None,verbose=True):
    """This function reads a gpx file, convert the geometry
    to UTM coordinates, project its points onto the osm graph edges
    and add useful features (see chapter (I))

    Parameters
    ----------
    file_path : the path to the gpx file

    G_osm : an osm multigraph

    max_projection_distance : maximal distance from a gpx point
    to its projection on the osm graph edges for it to be kept

    crs : the coordinates reference system to convert the geometry,
    should always be equal to the G_osm crs

    

    Returns
    -------
    the geodataframe containing the navigation data

    """
    t1=time.time()
    navigation_df=get_points_from_activity(file_path)
    navigation_df=navigation_df.drop_duplicates(subset=['geometry'])
    t2=time.time()
    if verbose:
        print('reading file took %f s'%(t2-t1))
    
    if crs is None:
        crs=navigation_df.estimate_utm_crs()
        G_osm=ox.project_graph(G_osm,crs)
    navigation_df=navigation_df.to_crs(crs)
    t3=time.time()
    if verbose:
        print('estimating crs took %f s'%(t3-t2))


    edges,distances=ox.nearest_edges(G_osm,[pt.x for pt in navigation_df['geometry']],[pt.y for pt in navigation_df['geometry']],return_dist=True)
    navigation_df.loc[:,'edge']=edges
    navigation_df.loc[:,'projection_distance']=distances
    navigation_df=navigation_df[navigation_df.projection_distance<max_projection_distance].drop(columns=['projection_distance'])

    navigation_df.loc[:,'geometry']=navigation_df.apply(lambda x:nearest_points(x['geometry'],G_osm.get_edge_data(*x['edge'])['geometry'])[1],axis=1)
    navigation_df.loc[:,'edge_coordinate']=navigation_df.apply(lambda x:G_osm.get_edge_data(*x['edge'])['geometry'].project(x['geometry']),axis=1)
    t4=time.time()
    if verbose:
        print('projecting took  %f s'%(t4-t3))


    navigation_df=add_segments(navigation_df)
    t5=time.time()
    if verbose:
        print('adding segments took %f s'%(t5-t4))

    return navigation_df



#NAVIGATION GRAPH BUILDING
def add_edge(G_navigation,edge,X,Y,T,id_segment,file_path,**kwargs):
    """This function adds a segment, which is the datum of the road axis X,
    the elevation Y, the time T, the id of the segment and the activity 
    it comes from, to the navigation graph along with the edge data from
    the osm graph

    Parameters
    ----------
    G_navigation : the mutlidigraph containing the navigation data

    edge : the edge of the navigation graph to which we add the data

    X : the x coordinates of the segment, i.e. the road coordinate

    Y : the y coordinates of the segment, i.e. the elevation

    id_segment : the id of the segment

    file_path : the path to the gpx file

    

    """
    if not(edge in G_navigation.edges(keys=True)):
        G_navigation.add_edge(*edge,Xs=[],Ys=[],Ts=[],id_segments=[],file_paths=[])
    datum=G_navigation.get_edge_data(*edge)
    Xs,Ys,Ts,id_segments,file_paths=datum['Xs'],datum['Ys'],datum['Ts'],datum['id_segments'],datum['file_paths']
    Xs.append(X)
    Ys.append(Y)
    Ts.append(T)
    id_segments.append(id_segment)
    file_paths.append(file_path)
    nx.set_edge_attributes(G_navigation,{edge:{'Xs':Xs,'Ys':Ys,'id_segments':id_segments,'file_paths':file_paths,**kwargs}})
    
def build_multidigraph(G_osm,navigation_df):
    """This function builds the navigation multidigraph from the
    osm graph and the navigation dataframe.
    For each segment with corresponding edge (u,v,k), an
    edge is added from u to v if the segment is positively
    oriented or from v to u otherwise, with attributes storing the 
    segment data and taking the key k into account. 
    Segments with one point are added in both directions.




    Parameters
    ----------
    G_osm : an osm multigraph

    navigation_df : the geodataframe containing the navigation data

    Returns
    -------
    the multidigraph containing the navigation data
    

    """

    G_navigation=nx.MultiDiGraph()
    for (file_path,id_segment),df in navigation_df.groupby(['file_path','segment']):
        X,Y,T=df['edge_coordinate'],df['elevation'],df['time']
        X,Y,T=np.array(X),np.array(Y),np.array(T)
        edge=df.iloc[0]['edge']
        data=G_osm.get_edge_data(*edge)
        orientation=df.iloc[0]['orientation']
        if orientation!=0:
            if orientation==-1.:
                X=data['length']-X
                edge=(edge[1],edge[0],edge[2])
            add_edge(G_navigation,edge=edge,X=X,Y=Y,T=T,id_segment=id_segment,constant=False,file_path=file_path,**data)
        else:
            add_edge(G_navigation,edge=edge,X=X,Y=Y,T=T,id_segment=id_segment,constant=True,file_path=file_path,**data)
            X=data['length']-X
            edge=(edge[1],edge[0],edge[2])
            add_edge(G_navigation,edge=edge,X=X,Y=Y,T=T,id_segment=id_segment,constant=True,file_path=file_path,**data)
    nx.set_node_attributes(G_navigation,{node:G_osm.nodes()[node] for node in G_navigation.nodes()})
    return G_navigation

#META SEGMENTS BUILDING

def update_meta_segment(meta_segment,Xs,Ys,Ts,id_segments,file_paths,merged,total_length,
                        max_id_segment_gap=2,max_time_gap=100.,max_distance_gap=250.):
    """This function update a metasegment. A metasegment is merged 
    with a segment from the edge if they both come from the same activity, i.e.
    shared the same file_path attribute and they are both close in the sense that:
    the last point of the meta segment and the first point of the segment have both 
    their x axis coordinates and time coordinates close.



    Parameters
    ----------
    meta_segment : the metasegments to be updated

    Xs : the list of the x-axis coordinates of the different segments

    Ys : the list of the y-axis coordinates of the different segments

    Ts : the list of the time coordinates of the different segments

    id_segments : the id of the different segments

    file_paths : the gpx file of the different segments

    merged : a list keeping track of the segments that have already been merged


    max_id_segment_gap : maximum gap autorized between the id of the last component of a metasegment 
    and the id of the segment comning from the edge to be merged. Ideally, consecutive segments along 
    a path that should be merged have consecutive id segments but small projection errors can insert
    small fake segments in between them.

    max_time_gap : maximum time gap between the last point of a metasegment and the first point
    of a segment for them to be merged

    max_distance_gap : maximum distance gap between the last point of a metasegment and the first point
    of a segment for them to be merged

    Returns
    -------
    the updated metasegment
    

    """
    for j,(X,Y,T,id_segment,file_path) in enumerate(zip(Xs,Ys,Ts,id_segments,file_paths)):
        if file_path==meta_segment['file_path']:
            for index_gap in range(1,max_id_segment_gap+1):
                 if (not(merged[j]) and
                    id_segment==meta_segment['last_id_segment']+index_gap and 
                    (T[0]-meta_segment['T'][-1]).total_seconds()<max_time_gap and 
                    X[0]+total_length-meta_segment['X'][-1]<max_distance_gap):

                    meta_segment=add_segment_to_meta_segment(meta_segment,X,Y,T,id_segment,file_path,total_length)
                    merged[j]=True
                    break
    return meta_segment


def update_meta_segments(meta_segments,G_navigation,G_osm,edge,nodes_positions,
                        max_id_segment_gap=2,max_time_gap=100.,max_distance_gap=250.):
    """This function update the list of meta_segments. All the meta_segments are updated
    (see update_meta_segment) and the segments that weren't merged are added as a metasegment.


    Parameters
    ----------
    meta_segments : a list of metasegments

    G_navigation : the mutlidigraph containing the navigation data

    G_osm : an osm graph, used to access the length of the edge's geometry in case the edge
    does not appear in G_navigation which might be useful foor some test cases

    edge : an edge from G_navigation whose segments should be merged to the meta_segments 

    nodes_positions : the x axis coordinates of the nodes on the path obtained by concatenating
    the edges of the path when appplying get_meta_segments_along_path

    max_id_segment_gap : maximum gap autorized between the id of the last component of a metasegment 
    and the id of the segment comning from the edge to be merged. Ideally, consecutive segments along 
    a path that should be merged have consecutive id segments but small projection errors can insert
    small fake segments in between them.

    max_time_gap : maximum time gap between the last point of a metasegment and the first point
    of a segment for them to be merged

    max_distance_gap : maximum distance gap between the last point of a metasegment and the first point
    of a segment for them to be merged


    Returns
    -------
    the list of updated meta_segments
    

    """
    total_length=nodes_positions[-1]
    if edge in G_osm.edges(keys=True):
        length=G_osm.get_edge_data(*edge)['length']
    else:
        length=G_osm.get_edge_data(edge[1],edge[0],edge[2])['length']
    datum=G_navigation.get_edge_data(*edge)

    if datum is not None:
        Xs,Ys,Ts,id_segments,file_paths=datum['Xs'],datum['Ys'],datum['Ts'],datum['id_segments'],datum['file_paths']
        merged=[False]*len(Xs)

        for k,meta_segment in enumerate(meta_segments):
            meta_segments[k]=update_meta_segment(meta_segment,Xs,Ys,Ts,id_segments,file_paths,merged,total_length,
                                                 max_id_segment_gap=max_id_segment_gap,max_time_gap=max_time_gap,
                                                 max_distance_gap=max_distance_gap)

        for k,(X,Y,T,id_segment,file_path) in enumerate(zip(Xs,Ys,Ts,id_segments,file_paths)):
            if not(merged[k]):
                meta_segments.append({'X':X+total_length,'Y':Y,'T':T,
                    'first_id_segment':id_segment,'last_id_segment':id_segment,'file_path':file_path,'increasing':True})

    nodes_positions.append(total_length+length)
    return meta_segments,nodes_positions




def add_segment_to_meta_segment(meta_segment,X,Y,T,id_segment,file_path,total_length):
    """This function merge a metasegment with a segment.


    Parameters
    ----------
    meta_segment : the metasegment to be updated

    X : the x-axis coordinates of the segment

    Ys : the y-axis coordinates of the segment

    Ts : the time coordinates of the segment

    id_segment : the id of the segment

    file_paths : the gpx file of the segment

    total_length : the length on the current path along which
    we're building the meta_segments


    Returns
    -------
    the concatenation of the metasegment and the segment
    

    """
    X=np.concatenate([meta_segment['X'],X+total_length])
    Y=np.concatenate([meta_segment['Y'],Y])
    T=np.concatenate([meta_segment['T'],T])
    return {'X':X,'Y':Y,'T':T,'first_id_segment':meta_segment['first_id_segment'],'last_id_segment':id_segment,
    'file_path':file_path,'increasing':True}


def non_max_suppression(meta_segments):
    """This function cleans the meta_segments : some
    meta_segments might be sub-meta_segments of 
    other ones and we don't want redundant data.


    Parameters
    ----------
    meta_segments : the metasegments to be cleaned


    Returns
    -------
    the cleaned meta_segments
    

    """
    keep=[True]*len(meta_segments)
    for (k1,meta_segment_1),(k2,meta_segment_2)  in combinations(enumerate(meta_segments),2):
        if keep[k1] and keep[k2] and meta_segment_1['file_path']==meta_segment_2['file_path']:
            first_1,last_1=meta_segment_1['first_id_segment'],meta_segment_1['last_id_segment']
            first_2,last_2= meta_segment_2['first_id_segment'],meta_segment_2['last_id_segment']
            if first_1<=first_2 and last_1>=last_2:
                keep[k2]=False 
            if first_1>=first_2 and last_1<=last_2:
                keep[k1]=False 

    return [elem for k,elem in enumerate(meta_segments) if keep[k]]




def get_meta_segments_along_path(path,G_navigation,G_osm,
                                 max_id_segment_gap=2,max_time_gap=100.,max_distance_gap=250.):

    """This function update the list of meta_segments. All the meta_segments are updated
    (see update_meta_segment) and the segments that weren't merged are added as a metasegment.


    Parameters
    ----------

    path : a succession of edges from the navigation graph G_navigation, 
    they form a path along which we're consecutively merging edges segments.
    The same procedure in done following the reversed path to obtained more data.

    G_navigation : the mutlidigraph containing the navigation data

    G_osm : an osm graph, used to access the length of the edge's geometry

    edge : an edge from G_navigation whose segments should be merged to the meta_segments 

    max_id_segment_gap : maximum gap autorized between the id of the last component of a metasegment 
    and the id of the segment comning from the edge to be merged. Ideally, consecutive segments along 
    a path that should be merged have consecutive id segments but small projection errors can insert
    small fake segments in between them.

    max_time_gap : maximum time gap between the last point of a metasegment and the first point
    of a segment for them to be merged

    max_distance_gap : maximum distance gap between the last point of a metasegment and the first point
    of a segment for them to be merged


    Returns
    -------
    the list of meta_segments along the path
    

    """
    meta_segments_forwards,nodes_positions=[],[0]
    for edge in path:
        meta_segments_forwards,nodes_positions=update_meta_segments(meta_segments_forwards,G_navigation,G_osm,edge,nodes_positions,
                                                     max_id_segment_gap=max_id_segment_gap,max_time_gap=max_time_gap,max_distance_gap=max_distance_gap)

    path=[(edge[1],edge[0],edge[2]) for edge in path[::-1]]
    meta_segments_backwards,reversed_nodes_positions=[],[0]
    for edge in path:
        meta_segments_backwards,reversed_nodes_positions=update_meta_segments(meta_segments_backwards,G_navigation,G_osm,edge,reversed_nodes_positions,
                                                     max_id_segment_gap=max_id_segment_gap,max_time_gap=max_time_gap,max_distance_gap=max_distance_gap)

    tot_length=nodes_positions[-1]
    for elem in meta_segments_backwards:
        elem.update({'X':tot_length-elem['X'][::-1],'Y':elem['Y'][::-1],'T':elem['T'][::-1],'increasing':False})

    return non_max_suppression(meta_segments_backwards+meta_segments_forwards),nodes_positions


def split_meta_segments(meta_segments,max_delta_T=900,overlap_coeff=0.):
    """This function split the metasegments that with a too big time amplitude.
    This might seem contradictory to split them after having done everything to 
    merge them but, since we will be trying to estimate the error in the axis 
    coordinate which change as time goes we need to do so. 
    Also, since the different metasegments have different underlying speeds we 
    might hope that the splitting points appear at different places and that we 
    still have a lot of overlapping curves. 
    If it's not the case, we can still extend the splitted metasegments in time
    with overlap_coeff at the cost of having a few redundant data.


    Parameters
    ----------

    meta_segments : the metasegments to be split

    max_delta_T : the maximum time amplitude accepted for a metasegment

    overlap_coeff : splitted curves are extended in time by overlap_coeff*max_delta_T

    Returns
    -------
    the list of possibly splitted metasegments
    

    """
    splitted_meta_segments=[]
    for elem in meta_segments:
        X,Y,T,file_path=elem['X'],elem['Y'],elem['T'],elem['file_path']
        x_min,x_max=np.min(X),np.max(X)
        dT=np.array([(t-T[0]).total_seconds() for t in T])
        delta=abs(dT[-1])

        if delta<max_delta_T:
            splitted_meta_segments.append({'file_path':file_path,'X':X,'Y':Y,'x_min':x_min,'x_max':x_max})
        else:
            N=round(delta/max_delta_T)
            max_delta_T=delta/N
            cuts=np.linspace(np.min(dT),np.max(dT),N+1)
            for k in range(len(cuts)-1):
                indexes=np.where((dT>=cuts[k]-overlap_coeff*max_delta_T)&(dT<=cuts[k+1]+overlap_coeff*max_delta_T))[0]
                if len(indexes)>0:
                    x,y=X[indexes],Y[indexes]
                    x_min,x_max=np.min(x),np.max(x)
                    splitted_meta_segments.append({'file_path':file_path,'X':x,'Y':y,'x_min':x_min,'x_max':x_max})

    return splitted_meta_segments


#SHIFT COMPUTATIONS
def affine(x1,x2,y1,y2):
    """This function return the affine functions that
    takes the y1 value at the x1 point and the y2 value
    at the x2 point


    Parameters
    ----------

    x1 : x coordinate of the first point

    y1 : y coordinate of the first point

    x2 : x coordinate of the second point

    y2 : y coordinate of the second point


    Returns
    -------
    the affine function
    

    """
    def f(x):
        return y1+(x-x1)*(y2-y1)/(x2-x1)
    return f



def get_piecewise(X_harmonized,X,Y,x_min,x_max):
    """This function computes the piecewise linear
    approximation of a metasegment with a common
    x-axis coordinates X_harmonized.

    Parameters
    ----------

    X_harmonized : the common x-axis coordinates for all meta_segments

    X : the x coordinates of the metasegment

    Y : the y coordinates of the metasegment

    x_min : minimum of X

    x_max : maximum of X


    Returns
    -------
    k_min_har : the first index of the X_harmonized coordinates
    on which Y_har is defined 

    k_max_har : the last index of the X_harmonized coordinates
    on which Y_har is defined 

    Y_harmonized : the piecewise linear interpolation defined 
    (possibly partially) on X_harmonized
    

    """
    conds=[(X_harmonized>=X[i])&(X_harmonized<X[i+1]) for i in range(len(X)-1)]
    funcs=[affine(X[i],X[i+1],Y[i],Y[i+1]) for i in range(len(X)-1)]+[None]
    k_min_har=np.where(X_harmonized>=x_min)[0][0]
    k_max_har=np.where(X_harmonized<=x_max)[0][-1]
    Y_harmonized=np.piecewise(X_harmonized,conds,funcs)
    return k_min_har,k_max_har,Y_harmonized




def harmonize_meta_segments(meta_segments,harmonizing_step=5.):
    """This function add new data to the metasegments : X_harmonized
    and Y_harmonized. In order to compute the x-axis error, the different
    metasegments must be interpretated as real functions and one needs
    a common system of x-axis coordinates to compare them.
    To do so, we look at the minimum/maximum x axis coordinates appearing in all 
    metasegments and simply call np.arange(x_min,x_max,harmonizing_step)

    Parameters
    ----------

    meta_segments : the metasegments

    harmonizing_step : the step between consecutive x axis points in the
    common X_harmonized coordinates system. The greater it is, the faster
    and less acurrate the shifts computations will be.

    Returns
    -------

    the harmonized metasegments
    

    """
    harmonized_meta_segments=[]
    x_min=min([meta_segment['x_min'] for meta_segment in meta_segments])
    x_max=max([meta_segment['x_max'] for meta_segment in meta_segments])
    X_harmonized=np.arange(x_min,x_max,harmonizing_step)
    for k,meta_segment in enumerate(meta_segments):
        x_min,x_max,X,Y=meta_segment['x_min'],meta_segment['x_max'],meta_segment['X'],meta_segment['Y']
        if x_max-x_min>harmonizing_step:
            k_min_har,k_max_har,Y_harmonized=get_piecewise(X_harmonized,X,Y,x_min,x_max)
            meta_segment.update({'X_har':X_harmonized,'Y_harmonized':Y_harmonized,'k_min_har':k_min_har,'k_max_har':k_max_har})
            harmonized_meta_segments.append(meta_segment)
    return harmonized_meta_segments


def discard_outliers(meta_segments,discarding_threshold=30):
    """This function discards weird meta_segments, weird in the sense 
    that they are too far from the median meta_segment.

    Parameters
    ----------

    meta_segments : the metasegments to be filtered

    discarding_threshold : the mean distance from the median metasegment 
    above which a metasegment is discarded

    Returns
    -------

    the filtered metasegments
    

    """
    Ys=np.array([meta_segment['Y_harmonized'] for meta_segment in meta_segments])
    median_meta_segment=np.nanmedian(Ys,axis=0)
    good_indexes=np.where(np.nanmean(np.abs(Ys-median_meta_segment),axis=1)<=discarding_threshold)[0]
    return [meta_segments[k] for k in good_indexes]




def normalized_correlation(meta_segment_1,meta_segment_2,k_max):
    """This function computes the cosine between shifted versions 
    of meta_segment_1 and meta_segment_2.

    Parameters
    ----------

    meta_segment_1 : first_meta_segment

    meta_segment_2 : second meta_segment

    k_max : the maximum index shift to consider

    Returns
    -------

    correlations : the list of cosines between the shifted version,
    correlations[k]=cos(Y1,Y2[+k]) for k in [-k_max,k_max]

    k_max : eventually reduced k_max in case the first one
    was greater than half the length of Y1
    

    """
    Y1,Y2=meta_segment_1['Y_harmonized'],meta_segment_2['Y_harmonized']
    k1_min,k1_max=meta_segment_1['k_min_har'],meta_segment_1['k_max_har']
    k2_min,k2_max=meta_segment_2['k_min_har'],meta_segment_2['k_max_har']
    Y1,Y2=Y1[max(k1_min,k2_min):min(k1_max,k2_max)],Y2[max(k1_min,k2_min):min(k1_max,k2_max)]
    k_max=min(k_max,len(Y1)//2)
    correlations=[]
    for k in range(k_max):
        y1,y2=Y1[k_max-k:],Y2[:k-k_max]
        norms=np.linalg.norm(y1)*np.linalg.norm(y2)
        if norms==0.:
            correlations.append(1.)
        else:
            correlations.append(np.sum(y1*y2)/norms)
    y1,y2=Y1,Y2
    norms=np.linalg.norm(y1)*np.linalg.norm(y2)
    if norms==0.:
        correlations.append(1.)
    else:
        correlations.append(np.sum(y1*y2)/norms)
    for k in range(1,k_max+1):
        y1,y2=Y1[:-k],Y2[k:]
        norms=np.linalg.norm(y1)*np.linalg.norm(y2)
        if norms==0.:
            correlations.append(1.)
        else:
            correlations.append(np.sum(y1*y2)/norms)
    return correlations,k_max



def is_quasi_affine(meta_segment,affine_threshold=2.5):
    """This function returns True if the meta_segment is
    close to be affine. If such it should not be taken into
    account whenc computing the shift.

    Parameters
    ----------

    meta_segment : a metasegment

    affine_threshold : a metasegment is considered quasi-affine if 
    its mean distance from its affine interpolation is below this threshold


    Returns
    -------

    True if the metasegment is close to be affine.
    

    """
    X,Y=meta_segment['X'],meta_segment['Y']
    Y_aff=Y[0]+(Y[-1]-Y[0])/(X[-1]-X[0])*(X-X[0])
    return np.mean(np.abs(Y-Y_aff))<affine_threshold


def get_shift(meta_segment_1,meta_segment_2,
              harmonizing_step=5.,max_shift_value=500.,
              overlay_threshold=0.25):
    """This function computes the shift in the x-axis between meta_segment_1 and meta_segment_2, 
    in the sense that adding this shift to the x coordinates of the meta_segments_1 points yields 
    a curve aligned with the one from meta_segments_2

    Parameters
    ----------

    meta_segment_1 : first_meta_segment

    meta_segment_2 : second meta_segment

    harmonizing_step : the step used to harmonized the slopes

    max_shift_value : the maximum possible shift that is likely to happend to reduce computations.

    overlay_threshold : the minimum overlapping ratio between the two meta_segments for the computation
    to happen. Metasegments merely overlapping can not be truthfuly realigned and we want to avoid
    useless computations.

    Returns
    -------

    overlay_1 : the ratio between the length of the overlapping interval and the definition interval of meta_segment_1

    overlay_2 : the ratio between the length of the overlapping interval and the definition interval of meta_segment_2

    shift : the shift between the two metasegments, eventually None if overlay_1 or overlay_2 are below the overlay_threshold
    

    """
    x1_min,x1_max,x2_min,x2_max=meta_segment_1['x_min'],meta_segment_1['x_max'],meta_segment_2['x_min'],meta_segment_2['x_max']
    x_min,x_max=max(x1_min,x2_min),min(x1_max,x2_max)
    overlay_1,overlay_2=(x_max-x_min)/(x1_max-x1_min),(x_max-x_min)/(x2_max-x2_min)
    if overlay_1<overlay_threshold and overlay_2<overlay_threshold:
        return overlay_1,overlay_2,None,None
    k_max=round(max_shift_value/harmonizing_step)
    corr,k_max=normalized_correlation(meta_segment_1,meta_segment_2,k_max)
    shift=(np.argmax(corr)-k_max)*harmonizing_step
    return overlay_1,overlay_2,shift,np.max(corr)


def get_pairwise_shifts(meta_segments,overlay_threshold=0.25,affine_threshold=2.5):
    """This function computes the shift between all pairs of meta_segments

    Parameters
    ----------

    meta_segments : the metasegments

    overlay_threshold : the minimum overlapping ratio between the two meta_segments for the computation
    to happen. Metasegments merely overlapping can not be truthfuly realigned and we want to avoid
    useless computations.

    affine_threshold : a metasegment is considered quasi-affine if its mean distance from its affine 
    interpolation is below this threshold


    Returns
    -------

    all the shifts between pairs of metasegemnts
    

    """
    affine_meta_segments=[]
    N=len(meta_segments)
    for k in range(N):
        if is_quasi_affine(meta_segments[k],affine_threshold=affine_threshold):
            affine_meta_segments.append(k)
    pairwise_shifts={}
    for k1,k2 in combinations(set(range(N))-set(affine_meta_segments),2):
        overlay_1,overlay_2,shift,corr=get_shift(meta_segments[k1],meta_segments[k2],overlay_threshold=overlay_threshold) 
        if shift is not None:
            pairwise_shifts[(k1,k2)]={'overlay':overlay_1,'shift':shift,'correlation':corr}
            pairwise_shifts[(k2,k1)]={'overlay':overlay_2,'shift':-shift,'correlation':corr}
    return pairwise_shifts,affine_meta_segments

def get_shifts_graph(pairwise_shifts,correlation_treshold=0.9):
    """This function computes the shift graph from the pairwise shifts

    Parameters
    ----------

    pairwise_shifts : the shifts between pairs of metasegments

    correlation_treshold : a shift computed with a correlation above this threshold
    will generate an edge between two metasegments


    Returns
    -------

    the graph encoding the shifts information
    

    """
    G=nx.DiGraph()
    for (k1,k2),d in pairwise_shifts.items():
        if d['correlation']>=correlation_treshold:
            G.add_edge(k1,k2,correlation=d['correlation'],shift=d['shift'],weight=-np.log(d['correlation'])-np.log(d['overlay']))

    return G



def realign_meta_segments_from_tree(shift_tree,meta_segments,min_tree_components=1):
    """This function realign each slopes by computing the estimating the absolute shift
    i.e. x axis error as the mean of its relative shifts with all other nodes (see documentation).


    Parameters
    ----------

    shift_tree : the maximum edge spanning tree

    meta_segments : the meta_segments to be realigned

    min_tree_components : the minimum size of a connected component in the shift tree 
    to estimate the absolute shift. If the size of the component is lower than
    this value, the underlying curves are discarded


    Returns
    -------

    the realigned metasegments
    

    """
    corrected_meta_segments=[]
    tree_un=nx.Graph(shift_tree)
    for cc in nx.connected_components(tree_un):
        N=len(cc)
        if N>=min_tree_components:
            all_paths=nx.shortest_path(nx.subgraph(tree_un,cc))
            for node,paths in all_paths.items():
                absolute_shift=0
                for path in paths.values():
                    absolute_shift-=sum([shift_tree.get_edge_data(path[i],path[i+1])['shift'] for i in range(len(path)-1)])
                absolute_shift/=N
                meta_segments[node].update({'X':meta_segments[node]['X']-absolute_shift,
                    'x_min':meta_segments[node]['x_min']-absolute_shift,'x_max':meta_segments[node]['x_max']-absolute_shift})
                corrected_meta_segments.append(meta_segments[node])
    return corrected_meta_segments



#SUB-METASEGMENTS PARTITION
def get_cover(meta_segments,min_count=2):
    """This function partitionate the big 
    intervals on which the different are 
    partially defined such that,for each 
    sub_interval, at least min_count metasegments
    will be defined on it. They will be the core
    intervals on which we will estimate the
    elevation profile.

    Parameters
    ----------

    meta_segments : the meta segments

    min_count : the minimum number of 
    overlapping meta_segments over a interval
    for it to be considered


    Returns
    -------

    global_meta_segments_indexes  : a list of 
    set of indexes such that global_meta_segments_indexes[i]
    contains the indexes of overlapping curves for
    extremities[i]

    extremities : a list of tuple representing
    intervals on which we have at least
    min_count overlapping curves
    

    """
    starts=[(min(elem['X']),1,k) for k,elem in enumerate(meta_segments)]
    ends=[(max(elem['X']),-1,k) for k,elem in enumerate(meta_segments)]
    L=sorted(starts+ends,key=lambda x:x[0])
    pts,moves,indexes_meta_segments=zip(*L)
    cover=False
    current_meta_segments_indexes,global_meta_segments_indexes,extremities=set(),[],[]
    count=0
    for k,index_meta_segment in enumerate(indexes_meta_segments):
        if moves[k]==1:
            current_meta_segments_indexes.add(index_meta_segment)
            count+=1
        else:
            current_meta_segments_indexes.remove(index_meta_segment)
            count-=1
        if count>=min_count:
            if not(cover):
                cover=True
                global_meta_segments_indexes.append(current_meta_segments_indexes.copy())
                extremities.append([pts[k]])
            else:
                if not(index_meta_segment in global_meta_segments_indexes[-1]):
                    global_meta_segments_indexes[-1].add(index_meta_segment)
        else:
            if cover:
                extremities[-1].append(pts[k])
                cover=False
    return global_meta_segments_indexes,extremities




#ESTIMATE ELEVATION PROFILE

def get_intermediate_elevation(intermediate_points,meta_segments):
    """This function computes the intermediate elevations at some
    chosen intermediate points by computing the median of the metasegments

    Parameters
    ----------

    intermediate_points : the x-axis of the points to be evaluated

    meta_segments : the meta segments 



    Returns
    -------

    median elevations    

    """
    intermediate_elevations=[]
    for k,pos in enumerate(intermediate_points):
        intermediate_elevations.append([])
        for j,meta_segment in enumerate(meta_segments):
            if meta_segment['x_min']<pos<meta_segment['x_max']:
                X,Y=meta_segment['X'],meta_segment['Y']
                index=np.where(X<pos)[0][-1]
                x1,y1,x2,y2=X[index],Y[index],X[index+1],Y[index+1]
                y=affine(x1,x2,y1,y2)(pos)
                intermediate_elevations[-1].append(y)
    return intermediate_elevations


# def clean_signal(X,Y,grad_thresh=1.):
#     dYdX=np.diff(Y)/np.diff(X)
#     pre_indexes=np.where(np.abs(dYdX)<=grad_thresh)[0]
#     indexes=list(sorted(set(pre_indexes).intersection(pre_indexes+1)))
#     return X[indexes],Y[indexes]


def approximate_derivative(meta_segments,x_min=None,x_max=None,
                           min_samples_leaf=25,min_impurity_decrease=0.5*float('1e-6')):
    """This function approximates the derivative of the elevation profile.
    We compute the pointwise derivative for each metasegment, combine them all together
    and sort them by the x coordinates and feed them to a decision tree regressor.


    Parameters
    ----------

    meta_segments : the metasegments

    x_min : the start of the interval on which we estimate the elevation profile

    x_max : the end of the interval on which we estimate the elevation profile

    min_samples_leaf : min_samplees_leaf for the DecisionTreeRegressor object

    min_impurity_decrease : min_impurity_decrease for the DecisionTreeRegressor object



    Returns
    -------

    X : the x coordinates of the derivatives

    dYdX : the value of the derivatives

    model : the trained DecisionTreeRegressor object    

    """
    all_derivatives=[]
    for k,meta_segment in enumerate(meta_segments):
        X,Y=meta_segment['X'],meta_segment['Y']
        if x_min is not None:
            indexes=np.where(X>=x_min)[0]
            X,Y=X[indexes],Y[indexes]
        if x_max is not None:
            indexes=np.where(X<=x_max)[0]
            X,Y=X[indexes],Y[indexes]
        # X,Y=clean_signal(X,Y,grad_thresh=1.)
        dX=np.diff(X)
        indexes=np.where(dX!=0.)[0]
        dYdX=np.diff(Y)[indexes]/dX[indexes]
        X=X[:-1]
        all_derivatives+=list(zip(X,dYdX))
    all_derivatives=sorted(list(all_derivatives),key=lambda x:x[0])
    if len(all_derivatives)==0:
        return None
    X,dYdX=list(zip(*all_derivatives))
    X,dYdX=np.array(X),np.array(dYdX)
    model=DecisionTreeRegressor(min_samples_leaf=min_samples_leaf,min_impurity_decrease=min_impurity_decrease)
    model.fit(X.reshape(-1,1),dYdX)
    return X,dYdX,model


def get_derivative_intervals(tree,node=0):
    """This function extracts the tree information
    in a more intelligible way that is the list of
    intervals on which the approximate derivative is 
    constant along with the value.

    Parameters
    ----------

    tree : the tree from the DecisionTreeRegressor object

    Returns
    -------

    a list of triplets (x1,x2,v) meaning that the approximate derivative is constant 
    on the intervals [x1,x2] with value v 

    """
    if tree.feature[node] == _tree.TREE_UNDEFINED:
        return [[-np.inf,np.inf,tree.value[node][0][0]]]
    else:
        threshold = tree.threshold[node]
        res_1,res_2=get_derivative_intervals(tree,tree.children_left[node]),get_derivative_intervals(tree,tree.children_right[node])
        res_1[-1][1]=threshold
        res_2[0][0]=threshold
        return res_1+res_2





def adjust_curve_elevation(Y,delta_expected):
    """This function perturbates a curve Y 
    represented by a sequence of elevation values
    so that the resulting curve is close to Y
    but have an elevation difference between the
    last and first point equal to the expected
    elevation difference delta_expected.


    Parameters
    ----------

    Y : the curve to be corrected

    delta_expected : the expected elevation difference


    Returns
    -------

    the perturbated curve




    """
    dY=np.diff(Y)
    dY_pos=np.where(dY>=0,dY,0)
    dY_neg=np.where(dY<0,-dY,0)
    delta_pos,delta_neg=np.sum(dY_pos),np.sum(dY_neg)
    if delta_pos==delta_neg==0:
        return Y,0.
    alpha=alpha=(delta_expected-(delta_pos-delta_neg))/(delta_pos+delta_neg)
    dY=(1+alpha)*dY_pos-(1-alpha)*dY_neg
    Y_pertubated=np.insert(np.cumsum(dY),0,0)
    return Y_pertubated+Y[0],delta_pos-delta_neg

def infer_curve_from_estimated_gradient(intervals,init_elevation=0):
    """This function integrates the piecewise constant derivate
    encoded by the intervals starting at init_elevation


    Parameters
    ----------

    intervals : a sequence of triplets (x1,x2,v) modelizing
    the derivate as a piecewise constant function which equals
    to v on [x1,x2].

    init_elevation : the starting elevation


    Returns
    -------

    the integrated curve

    """
    X,Y=[],[init_elevation]
    for x1,x2,alpha in intervals :
        X.append(x1)
        delta=(x2-x1)
        Y.append(Y[-1]+alpha*delta)
    X.append(x2)
    return X,Y

def approximate_elevation_profile(meta_segments,x_min,x_max,intermediate_distance=1000,
                                 min_samples_leaf=25,min_impurity_decrease=0.25*float('1e-6')):

    """This function estimates the elevation profile on the [x_min,x_max] intervals
    through the metasegments, it combines a pointwise estimation of some intermediate points
    at some  by computing the median and an estimation of the derivative between those 
    intermediate points.


    Parameters
    ----------

    meta_segments : the metasegments

    x_min : the start of the interval on which we estimate the elevation profile

    x_max : the end of the interval on which we estimate the elevation profile

    intermediate_distance : the distance used to sampled intermediate points at which
    we compute the median elevations. The greater it is, the more the final curve will
    be more C0 close but less C1 close to the real elevation profile.

    min_samples_leaf : min_samplees_leaf for the DecisionTreeRegressor object

    min_impurity_decrease : min_impurity_decrease for the DecisionTreeRegressor object


    Returns
    -------

    the integrated curve

    """

    intermediate_points=np.linspace(x_min,x_max,max(round((x_max-x_min)/intermediate_distance)+1,2))
    intermediate_elevations=get_intermediate_elevation(intermediate_points,meta_segments)
    X,Y=[],[]


    for k in range(len(intermediate_points)-1):
        x1,x2=intermediate_points[k],intermediate_points[k+1]
        output=approximate_derivative(meta_segments,x_min=x1,x_max=x2,
                                      min_samples_leaf=min_samples_leaf,
                                      min_impurity_decrease=min_impurity_decrease)

        if output is not None:
            _,_,model=output
            intervals=get_derivative_intervals(model.tree_)
            intervals[0][0]=x1
            intervals[-1][1]=x2
            x,y=infer_curve_from_estimated_gradient(intervals,np.nanmedian(intermediate_elevations[k]))
            delta_expected=np.nanmedian(intermediate_elevations[k+1])-np.nanmedian(intermediate_elevations[k])
            y,_=adjust_curve_elevation(y,delta_expected)
            X+=x[:-1]
            Y+=list(y[:-1])
    X.append(x[-1])
    Y.append(y[-1])

    return X,Y

#ELEVATION DATA RETRIEVAL


def get_nodes_elevation(nodes_positions,X,Y,x_min,x_max):
    """This function estimates the elevations at nodes positions
    on the metasegment

    Parameters
    ----------

    nodes_positions : the x-axis coordinates of the nodes

    X : the x-axis coordinates of the estimated elevation profile

    Y : the x-axis coordinates of the estimated elevation profile

    x_min : minimum X value

    x_max : maximum X value



    Returns
    -------

    a dictionary nodes_elevation with nodes_elevations[k]=estimated elevation
    for the node at nodes_positions[k]    

    """
    nodes_elevations={}
    for k,pos in enumerate(nodes_positions):
        if x_min<=pos<=x_max:
            index=np.where(X<=pos)[0][-1]
            x1,y1,x2,y2=X[index],Y[index],X[index+1],Y[index+1]
            elevation=affine(x1,x2,y1,y2)(pos)
            nodes_elevations[k]=elevation
    return nodes_elevations



def collect_elevation_information_from_sub_meta_segments(path,nodes_positions,meta_segments,x_min,x_max,
                                                         intermediate_distance=1000,min_samples_leaf=25,
                                                         min_impurity_decrease=0.25*float('1e-6')):

    """This function retrieves nodes and edges elevation data along a the portion of the path comprised between x_min and x_max
    in the navigation graph.
    Given a path, we estimate the elevation profile. Knowing the nodes positions (their x-axis coordinates along the road), 
    we can evaluate the elevation profile curve at these points to estimate the nodes elevations and also look at the restriction 
    of the elevation profile between two successive nodes to get a elevation profile for the edge joining them.


    Parameters
    ----------

    path : a path in the navigation graph, i.e. a list of edges

    nodes_positions : the x axis of the nodes

    meta_segments : the metasegments obtained along the path

    x_min : the start of the interval on which we estimate the elevation profile

    x_max : the end of the interval on which we estimate the elevation profile

    intermediate_distance : the distance used to sampled intermediate points at which
    we compute the median elevations. The greater it is, the more the final curve will
    be more C0 close but less C1 close to the real elevation profile.

    min_samples_leaf : min_samplees_leaf for the DecisionTreeRegressor object

    min_impurity_decrease : min_impurity_decrease for the DecisionTreeRegressor object

    Returns
    -------

    nodes_data : a dictionary whose keys are nodes and values are lists of the
    different median elevations computed

    edges_data : a dictionary whose keys are edges and values are lists of (possibly partial)
    elevations profiles computed

    """


    nodes=[edge[0] for edge in path]+[path[-1][1]]
    nodes_data,edges_data={},{}

    output=approximate_elevation_profile(meta_segments,x_min,x_max,intermediate_distance=intermediate_distance,
                                         min_samples_leaf=min_samples_leaf,min_impurity_decrease=min_impurity_decrease)
    if output is not None:
        X,Y=output
        X,Y=np.array(X),np.array(Y)
        nodes_elevations=get_nodes_elevation(nodes_positions,X,Y,x_min,x_max)
        for node_index,elev in nodes_elevations.items():
            if not(nodes[node_index] in nodes_data.keys()):
                nodes_data[nodes[node_index]]=[]
            nodes_data[nodes[node_index]].append(elev)

        x_min,x_max=np.min(X),np.max(X)
        for k,edge in enumerate(path):
            start,end=nodes_positions[k],nodes_positions[k+1]
            if end>=x_min and start<=x_max:
                indexes=np.where((X>=start)&(X<=end))
                x,y=X[indexes],Y[indexes]
                if start>=x_min:
                    x=np.insert(x,0,start)
                    y=np.insert(y,0,nodes_elevations[k])
                if end<=x_max:
                    x=np.append(x,end)
                    y=np.append(y,nodes_elevations[k+1])
                x-=start
                if not(edge in edges_data.keys()):
                    edges_data[edge]=[]
                edges_data[edge].append({'X':x,'Y':y,'cover':(x[-1]-x[0])/(end-start),'length':x[-1]-x[0]})

        return nodes_data,edges_data


def collect_elevation_data_from_path(G_navigation,G_osm,path,
                                    max_id_segment_gap=3,max_time_gap=100.,max_distance_gap=250.,
                                    max_delta_T=900,overlap_coeff=0.,harmonizing_step=5.,
                                    discarding_threshold=30.,overlay_threshold=0.25,affine_threshold=2.5,
                                    correlation_treshold=0.99,min_tree_components=1,min_count=2,min_cover_length=500.,
                                    intermediate_distance=1000,min_samples_leaf=25,min_impurity_decrease=0.25*float('1e-6')):

    """This function retrieves nodes and edges elevation data along the path in the navigation graph.
    Given a path, we estimate the elevation profile. After retrieving the metasegments, processing them and
    aligning them, we split the intervals into sub intervals upon which one has enough data to estimate
    the elevation profile.

    Parameters
    ----------

    G_navigation : the mutlidigraph containing the navigation data

    G_osm : an osm graph, used to access the length of the edge's geometry

    path : a path in the navigation graph, i.e. a list of edges

    max_id_segment_gap : maximum gap autorized between the id of the last component of a metasegment and the 
    id of the segment comning from the edge to be merged. Ideally, consecutive segments along a path that should 
    be merged have consecutive id segments but small projection errors can insert small fake segments in between them.

    max_time_gap : maximum time gap between the last point of a metasegment and the first point
    of a segment for them to be merged

    max_distance_gap : maximum distance gap between the last point of a metasegment and the first point
    of a segment for them to be merged

    max_delta_T : the maximum time amplitude accepted for a metasegment

    overlap_coeff : splitted curves are extended in time by overlap_coeff*max_delta_T

    harmonizing_step : the step between consecutive x axis points in the common X_harmonized coordinates system. 
    The greater it is, the faster and less acurrate the shifts computations will be.

    discarding_threshold : the mean distance from the median metasegment above which a metasegment is discarded

    overlay_threshold : the minimum overlapping ratio between the two meta_segments for the computation to happen. 
    Metasegments merely overlapping can not be truthfuly realigned and we want to avoid useless computations.

    affine_threshold : a metasegment is considered quasi-affine if its mean distance from its affine interpolation is below this threshold

    correlation_treshold=0.9 : a shift computed with a correlation above this threshold will generate an edge between two metasegments

    min_tree_components : the minimum size of a connected component in the shift tree to estimate the absolute shift. 
    If the size of the component is lower than this value, the underlying curves are discarded

    min_count : the minimum number of overlapping meta_segments over a intervalfor it to be considered

    min_cover_length : the minimum length of a cover to approximate the elevation profile on it

    intermediate_distance : the distance used to sampled intermediate points at which we compute the median elevations. 
    The greater it is, the more the final curve will be more C0 close but less C1 close to the real elevation profile.

    min_samples_leaf : min_samplees_leaf for the DecisionTreeRegressor object

    min_impurity_decrease : min_impurity_decrease for the DecisionTreeRegressor object

    Returns
    -------

    nodes_data : a dictionary whose keys are nodes and values are lists of the
    different median elevations computed

    edges_data : a dictionary whose keys are edges and values are lists of (possibly partial)
    elevations profiles computed

    """




    meta_segments,nodes_positions=get_meta_segments_along_path(path,G_navigation,G_osm,
                                  max_id_segment_gap=max_id_segment_gap,max_time_gap=max_time_gap,
                                  max_distance_gap=max_distance_gap)

    meta_segments=split_meta_segments(meta_segments,max_delta_T=max_delta_T,overlap_coeff=overlap_coeff)
    if len(meta_segments)>0:
        meta_segments=harmonize_meta_segments(meta_segments,harmonizing_step=harmonizing_step)
        if len(meta_segments)>0:
            meta_segments=discard_outliers(meta_segments,discarding_threshold=discarding_threshold)
            if len(meta_segments)>0:
                pairwise_shifts,affine_meta_segments=get_pairwise_shifts(meta_segments,overlay_threshold=overlay_threshold,affine_threshold=affine_threshold)
                shift_G=get_shifts_graph(pairwise_shifts,correlation_treshold=correlation_treshold)
                edges=list(nx.minimum_spanning_edges(nx.Graph(shift_G),weight='weight',data=False))
                edges+=[(v,u) for u,v in edges]
                shift_tree=nx.edge_subgraph(shift_G,edges)
                corrected_meta_segments=realign_meta_segments_from_tree(shift_tree,meta_segments,min_tree_components=min_tree_components)
                corrected_meta_segments+=[meta_segments[k] for k in affine_meta_segments]
                if len(corrected_meta_segments)>0:
                    cover,extremities=get_cover(corrected_meta_segments,min_count=min_count)
                    all_nodes_data,all_edges_data={},{}
                    for i,(x_min,x_max) in enumerate(extremities):
                        if x_max-x_min>min_cover_length:
                            sub_meta_segments=[corrected_meta_segments[k] for k in cover[i]]
                            output=collect_elevation_information_from_sub_meta_segments(path,nodes_positions,meta_segments,x_min,x_max,
                                                             intermediate_distance=intermediate_distance,min_samples_leaf=min_samples_leaf,
                                                             min_impurity_decrease=min_impurity_decrease)
                            if output is not None:
                                nodes_data,edges_data=output
                                all_nodes_data.update(nodes_data)
                                all_edges_data.update(edges_data)
                    return all_nodes_data,all_edges_data







    
#GRAPH DECOMPOSITION

def pre_edge_score(G_navigation,edge_1,edge_2,max_id_segment_gap=3):
    """This function associates a pre_score to a pair of edges in the
    dual graph. The pre_score measures the number of activites for which 
    subsequent segments are likely to be merged in the get_meta_segments_along_path
    function when one crosses the oriented duel edge (edge_1,edge_2).


    Parameters
    ----------

    G_navigation : the mutlidigraph containing the navigation data

    edge_1 : first edge

    edge_2 : second edge

    max_id_segment_gap : maximum gap autorized between the id of the last 
    component of a metasegment and the id of the segment comning from the edge to be merged. 
    Ideally, consecutive segments along a path that should be merged have 
    consecutive id segments but small projection errors can insert small fake segments in between them.

    Returns
    -------

    the pre_score between edges

    """
    if not(edge_1 in G_navigation.edges(keys=True) and edge_2 in G_navigation.edges(keys=True)):
        return 0.
    datum_1,datum_2=G_navigation.get_edge_data(*edge_1),G_navigation.get_edge_data(*edge_2)
    L1=zip(datum_1['file_paths'],datum_1['id_segments'])
    L2=zip(datum_2['file_paths'],datum_2['id_segments'])
    return len(set([file_path_2 for (file_path_1,id_segment_1),(file_path_2,id_segment_2) in product(L1,L2) if file_path_1==file_path_2 and id_segment_1+1<=id_segment_2<=id_segment_1+max_id_segment_gap]))



def edge_score(G_navigation,edge_1,edge_2,max_id_segment_gap=3):
    """This function associates a score to a pair of edges by 
    computing the maximum of the pre_score between (edge_1,edge_2)
    and (edge_2,edge_1)

    Parameters
    ----------

    G_navigation : the mutlidigraph containing the navigation data

    edge_1 : first edge

    edge_2 : second edge

    max_id_segment_gap : maximum gap autorized between the id of the last 
    component of a metasegment and the id of the segment comning from the edge to be merged. 
    Ideally, consecutive segments along a path that should be merged have 
    consecutive id segments but small projection errors can insert small fake segments in between them.

    Returns
    -------

    the score between edges

    """
    return max(pre_edge_score(G_navigation,edge_1,edge_2,max_id_segment_gap=max_id_segment_gap),
               pre_edge_score(G_navigation,(edge_2[1],edge_2[0],edge_2[2]),(edge_1[1],edge_1[0],edge_1[2]),max_id_segment_gap=max_id_segment_gap))


def build_dual_graph(G_navigation,max_id_segment_gap=3):
    """This function computes the 
    dual graph of the navigation graph.
    The dual edges between two opposite
    edges are removed not to go backwards
    when generating a path in the dual graph.


    Parameters
    ----------

    G_navigation : the mutlidigraph containing the navigation data

    max_id_segment_gap : maximum gap autorized between the id of the last 
    component of a metasegment and the id of the segment comning from the edge to be merged. 
    Ideally, consecutive segments along a path that should be merged have 
    consecutive id segments but small projection errors can insert small fake segments in between them.

    Returns
    -------

    the dual graph

    """
    dual_G=nx.line_graph(G_navigation)
    dual_G=nx.DiGraph(dual_G)
    to_be_removed=[]
    for edge_1,edge_2 in dual_G.edges():
        (node_11,node_12,k1),(node_21,node_22,k2)=edge_1,edge_2
        if k1==k2 and node_11==node_22 and node_12==node_21:
            to_be_removed.append((edge_1,edge_2))
            to_be_removed.append((edge_2,edge_1))

    dual_G.remove_edges_from(to_be_removed)
    nx.set_edge_attributes(dual_G,{(edge_1,edge_2):{'score':edge_score(G_navigation,edge_1,edge_2,max_id_segment_gap=max_id_segment_gap)} for edge_1,edge_2 in dual_G.edges()}) 
    return dual_G


def build_dual_tree(dual_G):
    """This function computes
    the maximum spanning tree
    for the dual graph (maximum 
    regarding the dual edge socre)


    Parameters
    ----------

    G_navigation : the mutlidigraph containing the navigation data


    Returns
    -------

    the maximum spanning tree

    """
    tree_edges=nx.maximum_spanning_edges(nx.Graph(dual_G),data=False,weight='score')
    new_edges=[]
    for edge_1,edge_2 in tree_edges:
        if not( (edge_2,edge_1) in dual_G.edges() ):
            new_edges.append((edge_1,edge_2))
        elif not( (edge_1,edge_2) in dual_G.edges() ):
            new_edges.append((edge_2,edge_1))
        else:
            score_1,score_2=dual_G.get_edge_data(edge_1,edge_2)['score'],dual_G.get_edge_data(edge_2,edge_1)['score']
            if score_1<score_2:
                new_edges.append((edge_2,edge_1))
            else:
                new_edges.append((edge_1,edge_2))


    return nx.edge_subgraph(dual_G,new_edges)


def graph_decomposition(dual_tree):
    """This function returns a (quasi) partition
    of the dual edges in the dual tree.
    The longest path (regarding the score attribute)
    is computed in the tree and the edges components,
    along with their opposite, are deleted from 
    the tree until there is no edges.

    Parameters
    ----------

    dual_tree : the maximum spanning tree


    Returns
    -------

    a list of paths in the dual graph

    """
    paths,dual_tree_cop=[],dual_tree.copy()
    while(len(dual_tree_cop.nodes())>0):
        path=nx.dag_longest_path(dual_tree_cop, weight='score')
        paths.append(path)
        dual_tree_cop.remove_nodes_from(path+[(edge[1],edge[0],edge[2]) for edge in path])
    return paths
