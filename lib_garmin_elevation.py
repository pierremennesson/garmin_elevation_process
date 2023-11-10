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
def to_multi_graph(G_dir):
    new_edges=[]
    for u,v,k in G_dir.edges(keys=True):
        if not( (v,u,k) in new_edges):
            new_edges.append((u,v,k))
    return nx.edge_subgraph(G_dir,new_edges)

def add_missing_geometries(G,min_length=float('inf')):
    attr,to_be_removed={},[]
    for u,v,k,d in G.edges(data=True,keys=True):
        removed=False
        if not('geometry' in d):
            if 'length' in d:
                if d['length']<min_length:
                    ls=LineString([[G.nodes()[u]['x'],G.nodes()[u]['y']],[G.nodes()[v]['x'],G.nodes()[v]['y']]])
                    d.update({'geometry':ls})
                    attr[(u,v,k)]=d
                else:
                    removed=True
            else:
                removed=True
        if removed:
            to_be_removed.append((u,v,k))
    G.remove_edges_from(to_be_removed)
    nx.set_edge_attributes(G,attr)


# #GPX DATA COLLECTION
def chunk(file_paths,nb_cpu):
    N=len(file_paths)
    n=max(N//nb_cpu,1)
    L=[file_paths[i:i+n] for i in range(0,N,n)]
    if len(L)>nb_cpu:
        last=L[-1]
        L=L[:-1]
        for k,elem in enumerate(last):
            L[k].append(elem)
    return L


def get_points_from_activity(file_path):
    gpx_file = open(file_path, 'r')
    gpx = gpxpy.parse(gpx_file)
    try:
        data=gpx.tracks[0].segments[0].points
    except:
        data=gpx.waypoints
    return gpd.GeoDataFrame([{'file_path':file_path,'geometry':Point(pt.longitude,pt.latitude),'elevation':pt.elevation,'time':pt.time} for pt in data],geometry='geometry',crs="EPSG:4326")




# def add_segments(garmin_df):
#     edge_sequences_lengths=[len(list(g)) for _,g in groupby(list(garmin_df['edge']))]
#     indexes=[sum(edge_sequences_lengths[:i]) for i in range(len(edge_sequences_lengths)+1)]
#     segments=[i  for k in range(len(indexes)-1) for i in [k]*(indexes[k+1]-indexes[k])]
#     garmin_df['segment']=segments
#     return garmin_df


def add_segments(garmin_df):
    garmin_df['pre_segment']=None 
    garmin_df['orientation']=None 
    garmin_df['segment']=None 

    edge_sequences_lengths=[len(list(g)) for _,g in groupby(list(garmin_df['edge']))]
    indexes=[sum(edge_sequences_lengths[:i]) for i in range(len(edge_sequences_lengths)+1)]
    pre_segments=[i  for k in range(len(indexes)-1) for i in [k]*(indexes[k+1]-indexes[k])]
    garmin_df.loc[:,'pre_segment']=pre_segments
    garmin_df=garmin_df.drop_duplicates(subset=['pre_segment','distance_ls'])

    edge_sequences_lengths=[len(list(g)) for _,g in groupby(list(garmin_df['edge']))]
    indexes=[sum(edge_sequences_lengths[:i]) for i in range(len(edge_sequences_lengths)+1)]
    pre_segments=[i  for k in range(len(indexes)-1) for i in [k]*(indexes[k+1]-indexes[k])]
    garmin_df.loc[:,'pre_segment']=pre_segments

    for k in range(len(indexes)-1):
        edge_df=garmin_df.iloc[indexes[k]:indexes[k+1]]
        if len(edge_df)==1:
            edge_df.loc[:,'orientation']=0.
        else:
            orientation=np.sign(np.diff(edge_df['distance_ls']))
            edge_df.iloc[1:].loc[:,'orientation']=orientation
            edge_df.iloc[0:1].loc[:,'orientation']=orientation[0]
            
    oriented_edge_sequences_lengths=[len(list(g)) for _,g in groupby(list(zip(garmin_df['pre_segment'],garmin_df['orientation'])))]
    indexes=[sum(oriented_edge_sequences_lengths[:i]) for i in range(len(oriented_edge_sequences_lengths)+1)]
    segments=[i  for k in range(len(indexes)-1) for i in [k]*(indexes[k+1]-indexes[k])]
    garmin_df.loc[:,'segment']=segments
    return garmin_df


def merge_segments(garmin_df,min_df_ratio=0.1,delta_time=60):
    garmin_df['merge_segment']=None
    segment_sequences_lengths=[len(list(g)) for _,g in groupby(list(garmin_df['segment']))]
    indexes=[sum(segment_sequences_lengths[:i]) for i in range(len(segment_sequences_lengths)+1)]
    good_indexes_pairs=[[indexes[k],indexes[k+1],0] for k in range(len(indexes)-1) if indexes[k+1]-indexes[k]>=min_df_ratio*len(garmin_df[garmin_df.pre_segment==garmin_df.iloc[indexes[k]]['pre_segment']])]
    for k in range(len(good_indexes_pairs)-1):
        k1,k2=good_indexes_pairs[k][1]-1,good_indexes_pairs[k+1][0]
        row_1,row_2=garmin_df.iloc[k1],garmin_df.iloc[k2]
        if row_1['pre_segment']!=row_2['pre_segment'] or row_1['orientation']*row_2['orientation']==-1.0 or (row_2['time'] - row_1['time']).total_seconds()>delta_time:
            good_indexes_pairs[k+1][2]=good_indexes_pairs[k][2]+1
        else:
            good_indexes_pairs[k+1][2]=good_indexes_pairs[k][2]

    for k1,k2,merge_segment in good_indexes_pairs:
        garmin_df.iloc[k1:k2].loc[:,'merge_segment']=merge_segment
    return garmin_df.dropna(subset='merge_segment')







def preprocess(file_path,G,dist_max=20,crs=None):
    t1=time.time()
    garmin_df=get_points_from_activity(file_path)
    garmin_df=garmin_df.drop_duplicates(subset=['geometry'])
    t2=time.time()
    print('reading file took %f s'%(t2-t1))
    
    if crs is None:
        crs=garmin_df.estimate_utm_crs()
        G=ox.project_graph(G,crs)
    garmin_df=garmin_df.to_crs(crs)
    t3=time.time()
    print('estimating crs took %f s'%(t3-t2))


    edges,distances=ox.nearest_edges(G,[pt.x for pt in garmin_df['geometry']],[pt.y for pt in garmin_df['geometry']],return_dist=True)
    garmin_df.loc[:,'edge']=edges
    garmin_df.loc[:,'dist']=distances
    garmin_df=garmin_df[garmin_df.dist<dist_max].drop(columns=['dist'])

    garmin_df.loc[:,'geometry']=garmin_df.apply(lambda x:nearest_points(x['geometry'],G.get_edge_data(*x['edge'])['geometry'])[1],axis=1)
    garmin_df.loc[:,'distance_ls']=garmin_df.apply(lambda x:G.get_edge_data(*x['edge'])['geometry'].project(x['geometry']),axis=1)
    t4=time.time()
    print('projecting took  %f s'%(t4-t3))


    garmin_df=add_segments(garmin_df)
    # garmin_df=merge_segments(garmin_df)
    t5=time.time()
    print('adding segments took %f s'%(t5-t4))

    return garmin_df



#GARMIN GRAPH BUILDING
def add_edge(GG,edge,X,Y,T,id_segment,constant,file_path,**kwargs):

    if not(edge in GG.edges(keys=True)):
        GG.add_edge(*edge,Xs=[],Ys=[],Ts=[],id_segments=[],constants=[],file_paths=[])
    datum=GG.get_edge_data(*edge)
    Xs,Ys,Ts,id_segments,constants,file_paths=datum['Xs'],datum['Ys'],datum['Ts'],datum['id_segments'],datum['constants'],datum['file_paths']
    Xs.append(X)
    Ys.append(Y)
    Ts.append(T)
    id_segments.append(id_segment)
    constants.append(constant)
    file_paths.append(file_path)
    nx.set_edge_attributes(GG,{edge:{'Xs':Xs,'Ys':Ys,'id_segments':id_segments,'constants':constants,'file_paths':file_paths,**kwargs}})
    
def build_multidigraph(G_proj,garmin_df):
    GG=nx.MultiDiGraph()
    for (file_path,id_segment),df in garmin_df.groupby(['file_path','segment']):
        X,Y,T=df['distance_ls'],df['elevation'],df['time']
        X,Y,T=np.array(X),np.array(Y),np.array(T)
        edge=df.iloc[0]['edge']
        data=G_proj.get_edge_data(*edge)
        orientation=df.iloc[0]['orientation']
        if orientation!=0:
            if orientation==-1.:
                X=data['length']-X
                edge=(edge[1],edge[0],edge[2])
            add_edge(GG,edge=edge,X=X,Y=Y,T=T,id_segment=id_segment,constant=False,file_path=file_path,**data)
        else:
            add_edge(GG,edge=edge,X=X,Y=Y,T=T,id_segment=id_segment,constant=True,file_path=file_path,**data)
            X=data['length']-X
            edge=(edge[1],edge[0],edge[2])
            add_edge(GG,edge=edge,X=X,Y=Y,T=T,id_segment=id_segment,constant=True,file_path=file_path,**data)

    nx.set_node_attributes(GG,{node:G_proj.nodes()[node] for node in GG.nodes()})
    return GG

#META SEGMENTS BUILDING
# def update_meta_segments(meta_segments,GG,G_proj,edge,node_positions,max_index_gap=3,max_time_gap=6000.,max_distance_gap=50000.):
    

#     updated_meta_segments,total_length=meta_segments.copy(),node_positions[-1]
#     length=G_proj.get_edge_data(*edge)['length']
#     datum=GG.get_edge_data(*edge)
#     if datum is not None:
#         length=datum['length']
#         Xs,Ys,Ts,id_segments,file_paths=datum['Xs'],datum['Ys'],datum['Ts'],datum['id_segments'],datum['file_paths']
#         for X,Y,T,id_segment,file_path in zip(Xs,Ys,Ts,id_segments,file_paths):
#             if not(file_path in meta_segments.keys()):
#                 if not(file_path in updated_meta_segments.keys()):
#                     updated_meta_segments[file_path]=[]
#                 updated_meta_segments[file_path].append({'X':X+total_length,'Y':Y,'T':T,'last_id_segment':id_segment,'first_id_segment':id_segment})
#             else:
#                 L=[]
#                 merge=False
#                 for meta_segment in updated_meta_segments[file_path]:
#                     for index_gap in range(1,max_index_gap+1):
#                          if id_segment==meta_segment['last_id_segment']+index_gap:
#                             print(file_path,meta_segment['last_id_segment'],id_segment)
#                             if (T[0]-meta_segment['T'][-1]).total_seconds()<max_time_gap and X[0]+total_length-meta_segment['X'][-1]<max_distance_gap: 
#                                 meta_segment=merge_segments(X,Y,T,meta_segment,id_segment,total_length)
#                                 L.append(meta_segment)
#                                 merge=True
#                             break


#                     if not(merge):
#                         L.append(meta_segment)
#                 if not(merge):
#                      L.append({'X':X+total_length,'Y':Y,'T':T,'last_id_segment':id_segment,'first_id_segment':id_segment})
#                 updated_meta_segments[file_path]=L

#     node_positions.append(total_length+length)
#     return updated_meta_segments,node_positions





def update_meta_segments(meta_segments,GG,G_proj,edge,nodes_positions,max_index_gap=6,max_time_gap=100.,max_distance_gap=250.):
    

    total_length=nodes_positions[-1]
    if edge in G_proj.edges(keys=True):
        length=G_proj.get_edge_data(*edge)['length']
    else:
        length=G_proj.get_edge_data(edge[1],edge[0],edge[2])['length']
    datum=GG.get_edge_data(*edge)
    if datum is not None:
        Xs,Ys,Ts,id_segments,file_paths=datum['Xs'],datum['Ys'],datum['Ts'],datum['id_segments'],datum['file_paths']
        merged=[False]*len(Xs)
        for k,meta_segment in enumerate(meta_segments):
            for j,(X,Y,T,id_segment,file_path) in enumerate(zip(Xs,Ys,Ts,id_segments,file_paths)):
                if file_path==meta_segment['file_path']:
                    for index_gap in range(1,max_index_gap+1):
                         if id_segment==meta_segment['last_id_segment']+index_gap and (T[0]-meta_segment['T'][-1]).total_seconds()<max_time_gap and X[0]+total_length-meta_segment['X'][-1]<max_distance_gap:
                            meta_segment=merge_meta_segments(X,Y,T,id_segment,meta_segment,total_length,file_path)
                            merged[j]=True

                            break
            meta_segments[k]=meta_segment
        for k,(X,Y,T,id_segment,file_path) in enumerate(zip(Xs,Ys,Ts,id_segments,file_paths)):
            if not(merged[k]):
                meta_segments.append({'X':X+total_length,'Y':Y,'T':T,'first_id_segment':id_segment,'last_id_segment':id_segment,'file_path':file_path,'increasing':True})
    nodes_positions.append(total_length+length)
    return meta_segments,nodes_positions




def merge_meta_segments(X,Y,T,id_segment,meta_segment,total_length,file_path):
    X=np.concatenate([meta_segment['X'],X+total_length])
    Y=np.concatenate([meta_segment['Y'],Y])
    T=np.concatenate([meta_segment['T'],T])
    return {'X':X,'Y':Y,'T':T,'first_id_segment':meta_segment['first_id_segment'],'last_id_segment':id_segment,'file_path':file_path,'increasing':True}


def non_max_suppression(meta_segments):
    keep=[True]*len(meta_segments)
    for (k1,meta_segment_1),(k2,meta_segment_2)  in combinations(enumerate(meta_segments),2):
        if keep[k1] and keep[k2] and meta_segment_1['file_path']==meta_segment_2['file_path']:
            first_1,last_1,first_2,last_2=meta_segment_1['first_id_segment'],meta_segment_1['last_id_segment'],meta_segment_2['first_id_segment'],meta_segment_2['last_id_segment']
            if first_1<=first_2 and last_1>=last_2:
                keep[k2]=False 
            if first_1>=first_2 and last_1<=last_2:
                keep[k1]=False 

    return [elem for k,elem in enumerate(meta_segments) if keep[k]]




def get_meta_segments_along_path(path,GG,G_proj):
    meta_segments_forwards,nodes_positions=[],[0]
    for edge in path:
        meta_segments_forwards,nodes_positions=update_meta_segments(meta_segments_forwards,GG,G_proj,edge,nodes_positions)

    path=[(edge[1],edge[0],edge[2]) for edge in path[::-1]]
    meta_segments_backwards,reversed_nodes_positions=[],[0]
    for edge in path:
        meta_segments_backwards,reversed_nodes_positions=update_meta_segments(meta_segments_backwards,GG,G_proj,edge,reversed_nodes_positions)

    tot_length=nodes_positions[-1]
    for elem in meta_segments_backwards:
        elem.update({'X':tot_length-elem['X'][::-1],'Y':elem['Y'][::-1],'T':elem['T'][::-1],'increasing':False})

    #return meta_segments_backwards+meta_segments_forwards,nodes_positions
    return non_max_suppression(meta_segments_backwards+meta_segments_forwards),nodes_positions



#SHIFT
def aff(x1,x2,y1,y2):
    def f(x):
        return y1+(x-x1)*(y2-y1)/(x2-x1)
    return f

def get_piecewise(X_har,X,Y,x_min,x_max,default_value=None):
    conds=[(X_har>=X[i])&(X_har<X[i+1]) for i in range(len(X)-1)]
    funcs=[aff(X[i],X[i+1],Y[i],Y[i+1]) for i in range(len(X)-1)]+[default_value]
    k_min_har=np.where(X_har>=x_min)[0][0]
    k_max_har=np.where(X_har<=x_max)[0][-1]
    return k_min_har,k_max_har,np.piecewise(X_har,conds,funcs)


# def get_slopes(meta_segments,min_dist=0.,max_dist=float('inf'),overlap_coeff=0.1):
#     slopes=[]
#     for elem in meta_segments:
#         X,Y,file_path=elem['X'],elem['Y'],elem['file_path']
#         x_min,x_max=np.min(X),np.max(X)
#         delta=x_max-x_min
#         X,Y=np.array(X),np.array(Y)
#         if delta>min_dist:
#             if delta<1.5*max_dist:
#                 slopes.append({'file_path':file_path,'X':X,'Y':Y,'x_min':x_min,'x_max':x_max})
#             else:
#                 N=round(delta/max_dist)
#                 max_dist=delta/N
#                 cuts=np.linspace(x_min,x_max,N+1)
#                 for k in range(len(cuts)-1):
#                     indexes=np.where((X>=cuts[k]-overlap_coeff*max_dist)&(X<=cuts[k+1]+overlap_coeff*max_dist))[0]
#                     x,y=X[indexes],Y[indexes]
#                     x_min,x_max=np.min(x),np.max(x)
#                     slopes.append({'file_path':file_path,'X':x,'Y':y,'x_min':x_min,'x_max':x_max})

#     return slopes


def get_slopes(meta_segments,max_delta_T=900,overlap_coeff=0.1):
    slopes=[]
    for elem in meta_segments:
        X,Y,T,file_path=elem['X'],elem['Y'],elem['T'],elem['file_path']
        x_min,x_max=np.min(X),np.max(X)
        dT=np.array([(t-T[0]).total_seconds() for t in T])
        delta=abs(dT[-1])

        if delta<max_delta_T:
            slopes.append({'file_path':file_path,'X':X,'Y':Y,'x_min':np.min(X),'x_max':np.max(X)})
        else:
            N=round(delta/max_delta_T)
            max_delta_T=delta/N
            cuts=np.linspace(np.min(dT),np.max(dT),N+1)
            for k in range(len(cuts)-1):
                indexes=np.where((dT>=cuts[k]-overlap_coeff*max_delta_T)&(dT<=cuts[k+1]+overlap_coeff*max_delta_T))[0]
                if len(indexes)>0:
                    x,y=X[indexes],Y[indexes]
                    x_min,x_max=np.min(x),np.max(x)
                    slopes.append({'file_path':file_path,'X':x,'Y':y,'x_min':x_min,'x_max':x_max})

    return slopes


def harmonize_slopes(slopes,step=5.):
    new_slopes=[]
    x_min=min([slope['x_min'] for slope in slopes])
    x_max=max([slope['x_max'] for slope in slopes])
    X_har=np.arange(x_min,x_max,step)
    for k,slope in enumerate(slopes):
        x_min,x_max,X,Y=slope['x_min'],slope['x_max'],slope['X'],slope['Y']
        if x_max-x_min>step:
            k_min_har,k_max_har,Y_har=get_piecewise(X_har,X,Y,x_min,x_max)
            slope.update({'X_har':X_har,'Y_har':Y_har,'k_min_har':k_min_har,'k_max_har':k_max_har})
            new_slopes.append(slope)
    return new_slopes


def discard_outliers(slopes,thresh=30):
    Ys=np.array([slope['Y_har'] for slope in slopes])
    median_slope=np.nanmedian(Ys,axis=0)
    good_indexes=np.where(np.nanmean(np.abs(Ys-median_slope),axis=1)<=thresh)[0]
    return [slopes[k] for k in good_indexes]




def normalized_correlation(slope_1,slope_2,k_max):
    X,Y=slope_1['Y_har'],slope_2['Y_har']
    k1_min,k1_max,k2_min,k2_max=slope_1['k_min_har'],slope_1['k_max_har'],slope_2['k_min_har'],slope_2['k_max_har']
    X,Y=X[max(k1_min,k2_min):min(k1_max,k2_max)],Y[max(k1_min,k2_min):min(k1_max,k2_max)]
    k_max=min(k_max,len(X)//2)
    res=[]
    for k in range(k_max):
        x,y=X[k_max-k:],Y[:k-k_max]
        # x,y=x-np.mean(x),y-np.mean(y)
        stds=np.linalg.norm(x)*np.linalg.norm(y)
        if stds==0.:
            res.append(1.)
        else:
            res.append(np.sum(x*y)/stds)
    # x,y=X-np.mean(X),Y-np.mean(Y)
    stds=np.linalg.norm(x)*np.linalg.norm(y)
    if stds==0.:
        res.append(1.)
    else:
        res.append(np.sum(x*y)/stds)
    for k in range(1,k_max+1):
        x,y=X[:-k],Y[k:]
        # x,y=x-np.mean(x),y-np.mean(y)
        stds=np.linalg.norm(x)*np.linalg.norm(y)
        if stds==0.:
            res.append(1.)
        else:
            res.append(np.sum(x*y)/stds)
    return res,k_max



def is_quasi_affine(slope,thresh=2.5):
    X,Y=slope['X'],slope['Y']
    Y_aff=Y[0]+(Y[-1]-Y[0])/(X[-1]-X[0])*(X-X[0])
    return np.mean(np.abs(Y-Y_aff))<thresh


def get_shift(slope_1,slope_2,step=5.,dist_max=500.,overlay_thresh=0.25):
    x1_min,x1_max,x2_min,x2_max=slope_1['x_min'],slope_1['x_max'],slope_2['x_min'],slope_2['x_max']
    x_min,x_max=max(x1_min,x2_min),min(x1_max,x2_max)
    overlay_1,overlay_2=(x_max-x_min)/(x1_max-x1_min),(x_max-x_min)/(x2_max-x2_min)
    if overlay_1<overlay_thresh and overlay_2<overlay_thresh:
        return overlay_1,overlay_2,None,None
    k_max=int(dist_max/step)
    corr,k_max=normalized_correlation(slope_1,slope_2,k_max)
    shift=(np.argmax(corr)-k_max)*step
    return overlay_1,overlay_2,shift,np.max(corr)


def get_pairwise_shifts(slopes,overlay_thresh=0.25):
    affine_slopes=[]
    N=len(slopes)
    for k in range(N):
        if is_quasi_affine(slopes[k]):
            affine_slopes.append(k)
    sum_pairwise_shifts,sum_weights,nb_neighbor={k:0 for k in range(len(slopes))},{k:0 for k in range(len(slopes))},{k:0 for k in range(len(slopes))}
    pairwise_shifts={}
    for k1,k2 in combinations(set(range(N))-set(affine_slopes),2):

        overlay_1,overlay_2,shift,corr=get_shift(slopes[k1],slopes[k2],overlay_thresh=overlay_thresh) 
        if shift is not None:
            pairwise_shifts[(k1,k2)]={'overlay':overlay_1,'shift':shift,'correlation':corr}
            pairwise_shifts[(k2,k1)]={'overlay':overlay_2,'shift':-shift,'correlation':corr}

    return pairwise_shifts,affine_slopes

def get_shifts_graph(pairwise_shifts,N,corr_tresh=0.9):
    G=nx.DiGraph()
    for (k1,k2),d in pairwise_shifts.items():
        if d['correlation']>=corr_tresh:
            G.add_edge(k1,k2,correlation=d['correlation'],shift=d['shift'],weight=-np.log(d['correlation'])-np.log(d['overlay']))

    return G



def realign_slopes_from_tree(shift_tree,slopes,min_components=1):
    corrected_slopes=[]
    shifts={}
    tree_un=nx.Graph(shift_tree)
    for cc in nx.connected_components(tree_un):
        N=len(cc)
        if N>=min_components:
            all_paths=nx.shortest_path(nx.subgraph(tree_un,cc))
            for node,paths in all_paths.items():
                absolute_shift=0
                for path in paths.values():
                    absolute_shift-=sum([shift_tree.get_edge_data(path[i],path[i+1])['shift'] for i in range(len(path)-1)])
                absolute_shift/=N
                slopes[node].update({'X':slopes[node]['X']-absolute_shift,'x_min':slopes[node]['x_min']-absolute_shift,'x_max':slopes[node]['x_max']-absolute_shift})
                corrected_slopes.append(slopes[node])
    return corrected_slopes,shifts



#SUB-SLOPES PARTITION
def get_cover(slopes,min_count=2):
    starts=[(min(elem['X']),1,k) for k,elem in enumerate(slopes)]
    ends=[(max(elem['X']),-1,k) for k,elem in enumerate(slopes)]
    L=sorted(starts+ends,key=lambda x:x[0])
    pts,moves,ids=zip(*L)
    cover=False
    l,L,extremities=set(),[],[]
    count=0
    for k,id_slope in enumerate(ids):
        if moves[k]==1:
            l.add(id_slope)
            count+=1
        else:
            l.remove(id_slope)
            count-=1
        if count>=min_count:
            if not(cover):
                cover=True
                L.append(l.copy())
                extremities.append([pts[k]])
            else:
                if not(id_slope in L[-1]):
                    L[-1].add(id_slope)
        else:
            if cover:
                extremities[-1].append(pts[k])
                cover=False
    return L,extremities




#ELEVATION DATA COLLECTION

####HANDLE INTERPOLATION DIST
def get_intermediate_elevation(intermediate_points,sub_slopes,max_interpolation_dist=100):
    intermediate_elevations=[]
    for k,pos in enumerate(intermediate_points):
        intermediate_elevations.append([])
        for j,slope in enumerate(sub_slopes):
            if slope['x_min']<pos<slope['x_max']:
                X,Y=slope['X'],slope['Y']
                index=np.where(X<pos)[0][-1]
                x1,y1,x2,y2=X[index],Y[index],X[index+1],Y[index+1]
                y=aff(x1,x2,y1,y2)(pos)
                intermediate_elevations[-1].append(y)
    return intermediate_elevations


def clean_signal(X,Y,grad_thresh=1.):
    dYdX=np.diff(Y)/np.diff(X)
    pre_indexes=np.where(np.abs(dYdX)<=grad_thresh)[0]
    indexes=list(sorted(set(pre_indexes).intersection(pre_indexes+1)))
    return X[indexes],Y[indexes]


def approximate_derivative(slopes,x_min=None,x_max=None,min_samples_leaf=50,min_impurity_decrease=0.5*float('1e-6'),criterion='squared_error'):
    all_derivatives=[]
    for k,slope in enumerate(slopes):
        X,Y=slope['X'],slope['Y']
        if x_min is not None:
            indexes=np.where(X>=x_min)[0]
            X,Y=X[indexes],Y[indexes]
        if x_max is not None:
            indexes=np.where(X<=x_max)[0]
            X,Y=X[indexes],Y[indexes]
        X,Y=clean_signal(X,Y,grad_thresh=1.)
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
    model=DecisionTreeRegressor(min_samples_leaf=min_samples_leaf,min_impurity_decrease=min_impurity_decrease,criterion=criterion)
    model.fit(X.reshape(-1,1),dYdX)
    return X,dYdX,model


def get_derivative_intervals(tree,node=0):
    if tree.feature[node] == _tree.TREE_UNDEFINED:
        return [[-np.inf,np.inf,tree.value[node][0][0]]]
    else:
        threshold = tree.threshold[node]
        res_1,res_2=get_derivative_intervals(tree,tree.children_left[node]),get_derivative_intervals(tree,tree.children_right[node])
        res_1[-1][1]=threshold
        res_2[0][0]=threshold
        return res_1+res_2

def simplify_tree(tree,x_min,x_max,min_interval=100,node=0):
    if tree.feature[node] == _tree.TREE_UNDEFINED or x_max-x_min<min_interval:
        return [[x_min,x_max,tree.value[node][0][0]]]
    else:
        threshold = tree.threshold[node]
        res_1,res_2=simplify_tree(tree,x_min,threshold,min_interval=min_interval,node=tree.children_left[node]),simplify_tree(tree,threshold,x_max,min_interval=min_interval,node=tree.children_right[node])
        if threshold-x_min<min_interval/2:
            (x1,x2,v1),(xx1,xx2,v2)=res_1[-1],res_2[0]
            v=(v1*(x2-x1)+v2*(xx2-xx1))/(xx2-x1)
            res_2[0]=[x1,xx2,v]
            return res_2
        elif x_max-threshold<min_interval/2:
            (x1,x2,v1),(xx1,xx2,v2)=res_1[-1],res_2[0]
            v=(v1*(x2-x1)+v2*(xx2-xx1))/(xx2-x1)
            res_1[-1]=[x1,xx2,v]
            return res_1
        else:
            return res_1+res_2





def approximate_slope(sub_slopes,x_min,x_max,intermediate_distance=1000):
    intermediate_points=np.linspace(x_min,x_max,max(round((x_max-x_min)/intermediate_distance)+1,2))
    intermediate_elevations=get_intermediate_elevation(intermediate_points,sub_slopes)
    output=approximate_derivative(sub_slopes,x_min=x_min,x_max=x_max,min_samples_leaf=25,min_impurity_decrease=0.25*float('1e-6'))
    if output is None:
        return None
    _,_,model=output
    intervals=simplify_tree(model.tree_,x_min,x_max,min_interval=100.)
    X,Y=[],[]
    for k in range(len(intermediate_points)-1):
        sub_intervals=[[min(max(x1,intermediate_points[k]),intermediate_points[k+1]),min(max(x2,intermediate_points[k]),intermediate_points[k+1]),v] for x1,x2,v in intervals if x2>=intermediate_points[k] and x1<=intermediate_points[k+1]]
        x,y=infer_curve_from_estimated_gradient(sub_intervals,np.nanmedian(intermediate_elevations[k]))
        delta_expected=np.nanmedian(intermediate_elevations[k+1])-np.nanmedian(intermediate_elevations[k])
        y,_=adjust_curve_elevation(y,delta_expected)
        X+=x
        Y+=list(y)
    X.append(x[-1])
    Y.append(y[-1])
    return X,Y


# def collect_elevation_information_from_sub_slopes(path,nodes_positions,sub_slopes,x_min,x_max):
#     nodes=[edge[0] for edge in path]+[path[-1][1]]
#     nodes_data,edges_data={},{}

#     output=approximate_slope(sub_slopes,x_min,x_max)
#     if output is not None:
#         X,Y=output

#         _,_,nodes_elevations=get_piecewise(nodes_positions,X,Y,x_min,x_max)
#         for k,elev in enumerate(nodes_elevations):
#             if elev==elev:
#                 if not(nodes[k] in nodes_data.keys()):
#                     nodes_data[nodes[k]]=[]
#                 nodes_data[nodes[k]].append(elev)


        # all_points=[[pos,nodes_elevations[k],k] for k,pos in enumerate(nodes_positions)]+[[x,y,-1] for x,y in zip(X,Y)]
        # all_points=sorted(all_points,key=lambda x:x[0])
        # all_points=np.array(all_points)
        # for k,edge in enumerate(path):
        #     if nodes_elevations[k]==nodes_elevations[k] and nodes_elevations[k+1]==nodes_elevations[k+1]:
        #         k1,k2=np.where(all_points[:,2]==k)[0][0],np.where(all_points[:,2]==k+1)[0][0]
        #         x=all_points[k1:k2+1,0]-all_points[k1,0]
        #         y=all_points[k1:k2+1,1]
        #         if not(edge in edges_data.keys()):
        #             edges_data[edge]=[]
        #         edges_data[edge].append({'X':x,'Y':y})

#         return nodes_data,edges_data

def collect_elevation_information_from_sub_slopes(path,nodes_positions,sub_slopes,x_min,x_max):
    nodes=[edge[0] for edge in path]+[path[-1][1]]
    nodes_data,edges_data={},{}

    output=approximate_slope(sub_slopes,x_min,x_max)
    if output is not None:
        X,Y=output
        X,Y=np.array(X),np.array(Y)
        _,_,nodes_elevations=get_piecewise(nodes_positions,X,Y,x_min,x_max)
        for k,elev in enumerate(nodes_elevations):
            if elev==elev:
                if not(nodes[k] in nodes_data.keys()):
                    nodes_data[nodes[k]]=[]
                nodes_data[nodes[k]].append(elev)

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
                edges_data[edge].append({'X':x,'Y':y})

        return nodes_data,edges_data


def collect_elevation_data_from_path(GG,G_proj,path):
    meta_segments,nodes_positions=get_meta_segments_along_path(path[:],GG,G_proj)
    slopes=get_slopes(meta_segments)
    if len(slopes)>0:
        slopes=harmonize_slopes(slopes)
        if len(slopes)>0:
            slopes=discard_outliers(slopes)
            if len(slopes)>0:
                pairwise_shifts,affine_slopes=get_pairwise_shifts(slopes,overlay_thresh=0.25,)
                shift_G=get_shifts_graph(pairwise_shifts,len(slopes),corr_tresh=0.99)
                edges=list(nx.minimum_spanning_edges(nx.Graph(shift_G),weight='weight',data=False))
                edges+=[(v,u) for u,v in edges]
                shift_tree=nx.edge_subgraph(shift_G,edges)
                corrected_slopes,shifts=realign_slopes_from_tree(shift_tree,slopes,min_components=1)
                corrected_slopes+=[slopes[k] for k in affine_slopes]
                if len(corrected_slopes)>0:
                    cover,extremities=get_cover(corrected_slopes,min_count=2)
                    all_nodes_data,all_edges_data={},{}
                    for i,(x_min,x_max) in enumerate(extremities):
                        sub_slopes=[corrected_slopes[k] for k in cover[i]]
                        output=collect_elevation_information_from_sub_slopes(path,nodes_positions,sub_slopes,x_min,x_max)
                        if output is not None:
                            nodes_data,edges_data=output
                            all_nodes_data.update(nodes_data)
                            all_edges_data.update(edges_data)
                    return all_nodes_data,all_edges_data

#COMBINING POINT AND DERIVATIVE ESTIMATES
def adjust_curve_elevation(Y,delta_expected):
    dY=np.diff(Y)
    dY_pos=np.where(dY>=0,dY,0)
    dY_neg=np.where(dY<0,-dY,0)
    delta_pos,delta_neg=np.sum(dY_pos),np.sum(dY_neg)
    alpha=alpha=(delta_expected-(delta_pos-delta_neg))/(delta_pos+delta_neg)
    dY=(1+alpha)*dY_pos-(1-alpha)*dY_neg
    Y_corr=np.insert(np.cumsum(dY),0,0)
    return Y_corr+Y[0],delta_pos-delta_neg

def infer_curve_from_estimated_gradient(intervals,init_elev=0):
    X,Y=[],[init_elev]
    for x1,x2,alpha in intervals :
        X.append(x1)
        delta=(x2-x1)
        Y.append(Y[-1]+alpha*delta)
    X.append(x2)
    return X,Y





    
#GRAPH DECOMPOSITION

def pre_edge_score(GG,edge_1,edge_2,max_index_gap=3):
    if not(edge_1 in GG.edges(keys=True) and edge_2 in GG.edges(keys=True)):
        return 0.
    datum_1,datum_2=GG.get_edge_data(*edge_1),GG.get_edge_data(*edge_2)
    L1=zip(datum_1['file_paths'],datum_1['id_segments'],datum_1['constants'])
    L2=zip(datum_2['file_paths'],datum_2['id_segments'],datum_2['constants'])
    return len(set([file_path_2 for (file_path_1,id_segment_1,constant_1),(file_path_2,id_segment_2,constant_2) in product(L1,L2) if file_path_1==file_path_2 and id_segment_1+1<=id_segment_2<=id_segment_1+max_index_gap]))



def edge_score(GG,edge_1,edge_2,max_index_gap=3):
    return max(pre_edge_score(GG,edge_1,edge_2,max_index_gap=max_index_gap),pre_edge_score(GG,(edge_2[1],edge_2[0],edge_2[2]),(edge_1[1],edge_1[0],edge_1[2]),max_index_gap=max_index_gap))


def build_dual_graph(GG):
    dual_G=nx.line_graph(GG)
    dual_G=nx.DiGraph(dual_G)
    to_be_removed=[]
    for edge_1,edge_2 in dual_G.edges():
        (node_11,node_12,k1),(node_21,node_22,k2)=edge_1,edge_2
        if k1==k2 and node_11==node_22 and node_12==node_21:
            to_be_removed.append((edge_1,edge_2))
            to_be_removed.append((edge_2,edge_1))

    dual_G.remove_edges_from(to_be_removed)
    nx.set_edge_attributes(dual_G,{(edge_1,edge_2):{'score':edge_score(GG,edge_1,edge_2)} for edge_1,edge_2 in dual_G.edges()}) 
    return dual_G


def build_dual_tree(dual_G):
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
    paths,dual_tree_cop=[],dual_tree.copy()
    while(len(dual_tree_cop.nodes())>0):
        path=nx.dag_longest_path(dual_tree_cop, weight='score')
        paths.append(path)
        dual_tree_cop.remove_nodes_from(path+[(edge[1],edge[0],edge[2]) for edge in path])
    return paths
