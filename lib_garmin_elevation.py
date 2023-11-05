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
import random


#OSM GRAPH CORRECTION
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
    garmin_df=merge_segments(garmin_df)
    t5=time.time()
    print('adding segments took %f s'%(t5-t4))

    return garmin_df



#GARMIN GRAPH BUILDING
def add_edge(GG,edge,X,Y,T,id_segment,file_path,**kwargs):

    if not(edge in GG.edges(keys=True)):
        GG.add_edge(*edge,Xs=[],Ys=[],Ts=[],id_segments=[],file_paths=[])
    datum=GG.get_edge_data(*edge)
    Xs,Ys,Ts,id_segments,file_paths=datum['Xs'],datum['Ys'],datum['Ts'],datum['id_segments'],datum['file_paths']
    Xs.append(X)
    Ys.append(Y)
    Ts.append(T)
    id_segments.append(id_segment)
    file_paths.append(file_path)
    nx.set_edge_attributes(GG,{edge:{'Xs':Xs,'Ys':Ys,'id_segments':id_segments,'file_paths':file_paths,**kwargs}})
    
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
            add_edge(GG,edge=edge,X=X,Y=Y,T=T,id_segment=id_segment,file_path=file_path,**data)
        else:
            add_edge(GG,edge=edge,X=X,Y=Y,T=T,id_segment=id_segment,file_path=file_path,**data)
            X=data['length']-X
            edge=(edge[1],edge[0],edge[2])
            add_edge(GG,edge=edge,X=X,Y=Y,T=T,id_segment=id_segment,file_path=file_path,**data)

    nx.set_node_attributes(GG,{node:G_proj.nodes()[node] for node in GG.nodes()})
    return GG

#META SEGMENTS BUILDING
def update_meta_segments(meta_segments,GG,G_proj,edge,node_positions,max_index_gap=1,max_time_gap=60.):
    

    updated_meta_segments,total_length=meta_segments.copy(),node_positions[-1]
    length=G_proj.get_edge_data(*edge)['length']
    datum=GG.get_edge_data(*edge)
    if datum is not None:
        length=datum['length']
        Xs,Ys,Ts,id_segments,file_paths=datum['Xs'],datum['Ys'],datum['Ts'],datum['id_segments'],datum['file_paths']
        for X,Y,T,id_segment,file_path in zip(Xs,Ys,Ts,id_segments,file_paths):
            if not(file_path in meta_segments.keys()):
                if not(file_path in updated_meta_segments.keys()):
                    updated_meta_segments[file_path]=[]
                updated_meta_segments[file_path].append({'X':X+total_length,'Y':Y,'T':T,'last_id_segment':id_segment,'first_id_segment':id_segment})
            else:
                L=[]
                merge=False
                for meta_segment in updated_meta_segments[file_path]:
                    for index_gap in range(1,max_index_gap+1):
                         if id_segment==meta_segment['last_id_segment']+index_gap:
                            if (T[0]-meta_segment['T'][-1]).total_seconds()<max_time_gap: 
                                X=np.concatenate([meta_segment['X'],X+total_length])
                                Y=np.concatenate([meta_segment['Y'],Y])
                                T=np.concatenate([meta_segment['T'],T])
                                L.append({'X':X,'Y':Y,'T':T,'first_id_segment':meta_segment['first_id_segment'],'last_id_segment':id_segment})
                                merge=True
                                break
                            else:
                                break

                    if not(merge):
                        L.append(meta_segment)
                if not(merge):
                     L.append({'X':X+total_length,'Y':Y,'T':T,'last_id_segment':id_segment,'first_id_segment':id_segment})
                updated_meta_segments[file_path]=L
        

    node_positions.append(total_length+length)
    return updated_meta_segments,node_positions


def merge_backwards_and_forwards(L_back,L_for):
    keep_back,keep_for=[True]*len(L_back),[True]*len(L_for)
    for (k1,meta_segment_back),(k2,meta_segment_for) in product(enumerate(L_back),enumerate(L_for)):
        if keep_back[k1] and keep_for[k2]:
            first_back,last_back,first_for,last_for=meta_segment_back['first_id_segment'],meta_segment_back['last_id_segment'],meta_segment_for['first_id_segment'],meta_segment_for['last_id_segment']
            if first_back<=first_for and last_back>=last_for:
                keep_for[k2]=False 
            elif first_back>=first_for and last_back<=last_for:
                keep_back[k1]=False 
    return [elem for k1,elem in enumerate(L_back) if keep_back[k1]]+[elem for k2,elem in enumerate(L_for) if keep_for[k2]]




def get_meta_segments_along_path(path,GG,G_proj):
    meta_segments_forwards,node_positions={},[0]
    for edge in path:
        meta_segments_forwards,node_positions=update_meta_segments(meta_segments_forwards,GG,G_proj,edge,node_positions)
    path=[(edge[1],edge[0],edge[2]) for edge in path[::-1]]
    meta_segments_backwards,reversed_nodes_positions={},[0]
    for edge in path:
        meta_segments_backwards,reversed_nodes_positions=update_meta_segments(meta_segments_backwards,GG,G_proj,edge,reversed_nodes_positions)
    tot_length=node_positions[-1]
    for file_path,L in meta_segments_backwards.items():
        for elem in L:
            elem.update({'X':tot_length-elem['X'][::-1],'Y':elem['Y'][::-1],'T':elem['T'][::-1]})
    res={}
    for file_path in set(meta_segments_forwards.keys()).union(set(meta_segments_backwards.keys())):
        L_for=[] if not(file_path in meta_segments_forwards.keys()) else meta_segments_forwards[file_path]
        L_back=[] if not(file_path in meta_segments_backwards.keys()) else meta_segments_backwards[file_path]
        res[file_path]=merge_backwards_and_forwards(L_back,L_for)

    return res,node_positions


#SHIFT
def aff(x1,x2,y1,y2):
    def f(x):
        return y1+(x-x1)*(y2-y1)/(x2-x1)
    return f

def get_piecewise(X_har,X,Y,x_min,x_max,default_value=None):
    conds=[(X_har>=X[i])&(X_har<X[i+1]) for i in range(len(X)-1)]
    funcs=[aff(X[i],X[i+1],Y[i],Y[i+1]) for i in range(len(X)-1)]+[default_value]
    x_min_har=np.where(X_har>=x_min)[0][0]
    x_max_har=np.where(X_har<=x_max)[0][-1]
    return x_min_har,x_max_har,np.piecewise(X_har,conds,funcs)


def get_slopes(meta_segments,min_dist=0.,max_dist=float('inf')):
    slopes=[]
    for file_path,L in meta_segments.items():
        for elem in L:
            X,Y=elem['X'],elem['Y']
            x_min,x_max=np.min(X),np.max(X)
            delta=x_max-x_min
            X,Y=np.array(X),np.array(Y)
            if delta>min_dist:
                if delta<1.5*max_dist:
                    slopes.append({'file_path':file_path,'X':X,'Y':Y,'x_min':x_min,'x_max':x_max})
                else:
                    N=round(delta/max_dist)
                    cuts=np.linspace(x_min,x_max,N+1)
                    for k in range(len(cuts)-1):
                        indexes=np.where((X>=cuts[k])&(X<=cuts[k+1]))[0]
                        slopes.append({'file_path':file_path,'X':X[indexes],'Y':Y[indexes],'x_min':cuts[k],'x_max':cuts[k+1]})

    return slopes

def harmonize_slopes(slopes,step=5.):
    x_min=min([slope['x_min'] for slope in slopes])
    x_max=max([slope['x_max'] for slope in slopes])
    X_har=np.arange(x_min,x_max,step)
    for slope in slopes:
        x_min,x_max,X,Y=slope['x_min'],slope['x_max'],slope['X'],slope['Y']
        x_min_har,x_max_har,Y_har=get_piecewise(X_har,X,Y,x_min,x_max)
        slope.update({'X_har':X,'Y_har':Y_har,'x_min_har':x_min_har,'x_max_har':x_max_har})
    return slopes

    


# def normalized_correlation(X,Y,k_max):
#     k_max=min(k_max,len(X)//2)
#     res=[]
#     for k in range(k_max):
#         x,y=X[k_max-k:],Y[:k-k_max]
#         x,y=x-np.mean(x),y-np.mean(y)
#         stds=np.linalg.norm(x)*np.linalg.norm(y)
#         if stds==0.:
#             res.append(1.)
#         else:
#             res.append(np.sum(x*y)/stds)
#     x,y=X-np.mean(X),Y-np.mean(Y)
#     stds=np.linalg.norm(x)*np.linalg.norm(y)
#     if stds==0.:
#         res.append(1.)
#     else:
#         res.append(np.sum(x*y)/stds)
#     for k in range(1,k_max+1):
#         x,y=X[:-k],Y[k:]
#         x,y=x-np.mean(x),y-np.mean(y)
#         stds=np.linalg.norm(x)*np.linalg.norm(y)
#         if stds==0.:
#             res.append(1.)
#         else:
#             res.append(np.sum(x*y)/stds)
#     return res,k_max


def normalized_correlation(slope_1,slope_2,k_max):
    X,Y=slope_1['Y_har'],slope_2['Y_har']
    k1_min,k1_max,k2_min,k2_max=slope_1['x_min_har'],slope_1['x_max_har'],slope_2['x_min_har'],slope_2['x_max_har']
    X,Y=X[max(k1_min,k2_min):min(k1_max,k2_max)],Y[max(k1_min,k2_min):min(k1_max,k2_max)]
    k_max=min(k_max,len(X)//2)
    res=[]
    for k in range(k_max):
        x,y=X[k_max-k:],Y[:k-k_max]
        x,y=x-np.mean(x),y-np.mean(y)
        stds=np.linalg.norm(x)*np.linalg.norm(y)
        if stds==0.:
            res.append(1.)
        else:
            res.append(np.sum(x*y)/stds)
    x,y=X-np.mean(X),Y-np.mean(Y)
    stds=np.linalg.norm(x)*np.linalg.norm(y)
    if stds==0.:
        res.append(1.)
    else:
        res.append(np.sum(x*y)/stds)
    for k in range(1,k_max+1):
        x,y=X[:-k],Y[k:]
        x,y=x-np.mean(x),y-np.mean(y)
        stds=np.linalg.norm(x)*np.linalg.norm(y)
        if stds==0.:
            res.append(1.)
        else:
            res.append(np.sum(x*y)/stds)
    return res,k_max



def is_quasi_affine(slope,thresh=2.):
    X,Y=slope['X'],slope['Y']
    Y_aff=Y[0]+(Y[-1]-Y[0])/(X[-1]-X[0])*(X-X[0])
    return np.mean(np.abs(Y-Y_aff))<thresh

# def get_shift(ref_pts,shifted_pts,step=5.,dist_max=500.,overlay_thresh=0.25):
#     (X1,Y1),(X2,Y2)=ref_pts,shifted_pts
#     x1_min,x1_max,x2_min,x2_max=min(X1),max(X1),min(X2),max(X2)
#     x_min,x_max=max(x1_min,x2_min),min(x1_max,x2_max)
#     overlay_1,overlay_2=(x_max-x_min)/(x1_max-x1_min),(x_max-x_min)/(x2_max-x2_min)
#     if overlay_1<overlay_thresh and overlay_2<overlay_thresh:
#         return overlay_1,overlay_2,None,None
#     X=np.arange(x_min,x_max,step)
#     if len(X)<2 or overlay_1 is None:
#         return overlay_1,overlay_2,None,None
#     YY1,YY2=get_piecewise(X,X1,Y1),get_piecewise(X,X2,Y2)
#     k_max=int(dist_max/step)
#     corr,k_max=normalized_correlation(YY1,YY2,k_max)
#     shift=(np.argmax(corr)-k_max)*step
#     return overlay_1,overlay_2,shift,np.max(corr)

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

def get_shifts_graph(pairwise_shifts,N,corr_tresh=0.98):
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
                slopes[node].update({'X':slopes[node]['X']-absolute_shift})

                corrected_slopes.append(slopes[node])
    return corrected_slopes,shifts



#PARTITION
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
def get_median_node_elevation(nodes_positions,sub_slopes,x_min,x_max,min_count=2):
    nodes_elevations,median_nodes_elevations={},{}
    for k,node_pos in enumerate(nodes_positions):
        if node_pos>=x_min and node_pos<=x_max:
            nodes_elevations[k]=[]
            for j,slope in enumerate(sub_slopes):
                X=slope['X']
                indexes=np.where(X<node_pos)[0]
                if len(indexes) not in [0,len(X)]:
                    Y=slope['Y']
                    index=indexes[-1]
                    x1,y1,x2,y2=X[index],Y[index],X[index+1],Y[index+1]
                    y=aff(x1,x2,y1,y2)(node_pos)
                    nodes_elevations[k].append(y)
    for k,l in nodes_elevations.items():
        if len(l)>=min_count:
            median_nodes_elevations[k]=np.median(l)

    return nodes_elevations,median_nodes_elevations



def clean_signal(X,Y,grad_thresh=1.):
    dYdX=np.diff(Y)/np.diff(X)
    pre_indexes=np.where(np.abs(dYdX)<=grad_thresh)[0]
    indexes=list(sorted(set(pre_indexes).intersection(pre_indexes+1)))
    return X[indexes],Y[indexes]


def approximate_derivative(slopes,x_min=None,x_max=None,min_interval=50.,min_samples_leaf=50,min_impurity_decrease=0.5*float('1e-6'),criterion='squared_error'):
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
        all_derivatives+=list(zip(X[:-1],dYdX))
    all_derivatives=sorted(list(all_derivatives),key=lambda x:x[0])
    if len(all_derivatives)==0:
        return None
    X,dYdX=list(zip(*all_derivatives))
    X,dYdX=np.array(X),np.array(dYdX)
    max_leaf_nodes=max(int((np.max(X)-np.min(X))/min_interval)+1,2)
    model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,min_samples_leaf=min_samples_leaf,min_impurity_decrease=min_impurity_decrease,criterion=criterion)
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



def collect_elevation_information_from_sub_slopes(path,nodes_positions,sub_slopes,x_min,x_max):
    _,median_nodes_elevations=get_median_node_elevation(nodes_positions,sub_slopes,x_min,x_max)
    median_nodes_elevations={path[k]:m for k,m in median_nodes_elevations.items()}

    X,dYdX,model=approximate_derivative(sub_slopes,x_min=x_min,x_max=x_max,min_samples_leaf=50,min_impurity_decrease=0.25*float('1e-6'))
    intervals=get_derivative_intervals(model.tree_)
    intervals[0][0]=x_min
    intervals[-1][1]=x_max
    edge_intervals={}
    for k in range(len(nodes_positions)-1):
        start,end=nodes_positions[k],nodes_positions[k+1]
        shifted_intervals=[(max(0,x1-start),min(x2-start,end-start),v) for x1,x2,v in intervals if x2>=start and x1<end]
        if len(shifted_intervals)>0:
            edge_intervals[(path[k],path[k+1])]=shifted_intervals
    return median_nodes_elevations,edge_intervals



#COMBINING POINT AND DERIVATIVE ESTIMATES
def adjust_curve_elevation(Y,delta_expected):
    dY=np.diff(Y)
    dY_pos=np.where(dY>=0,dY,0)
    dY_neg=np.where(dY<0,-dY,0)
    delta_pos,delta_neg=np.sum(dY_pos),np.sum(dY_neg)
    alpha=alpha=(delta_expected-(delta_pos-delta_neg))/(delta_pos+delta_neg)
    dY=(1+alpha)*dY_pos-(1-alpha)*dY_neg
    return np.insert(np.cumsum(dY),0,0),delta_pos-delta_neg

def infer_curve_from_estimated_gradient(intervals,init_elev=0):
    X,Y=[],[init_elev]
    for x1,x2,alpha in intervals :
        X.append(x1)
        delta=(x2-x1)
        Y.append(Y[-1]+alpha*delta)
    X.append(x2)
    return X,Y

# def get_final_slope(meta_segments,length,min_gap=float('inf')):
#     slopes=get_slopes(meta_segments)
#     pairwise_shifts=get_pairwise_shifts(slopes)
#     shift_G,affine_slopes=get_shifts_graph(pairwise_shifts,len(slopes))
#     corrected_slopes=realign_slopes_from_graph(shift_G,slopes)+[slopes[k] for k in affine_slopes]
#     if len(corrected_slopes)==0:
#         return None
#     if min(slope['X'][0] for slope in slopes)>min_gap or max(slope['X'][-1] for slope in slopes)<length-min_gap:
#         return None

#     output=approximate_derivative(corrected_slopes,length)
#     if output is None:
#         return None
#     _,_,model=output
#     intervals=recurse(model.tree_,0)
#     intervals[0][0]=0
#     intervals[-1][1]=length
#     X,Y=infer_curve_from_estimated_gradient(intervals)
#     return X,Y