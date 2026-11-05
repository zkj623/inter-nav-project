"""Offline task qualification; these maps must never seed the online navigator.

Tortuosity uses the continuous shortest path in the fixed raster-derived free
polygon (visibility graph), not the executed or clearance-weighted route.
The uninflated result is a lower bound for finite-size motion on the same
occupancy representation. A separate
0.30m clearance graph screens robot connectivity; endpoints need 0.60m clearance.
"""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
from scipy import ndimage,sparse
from scipy.sparse.csgraph import dijkstra
import shapely
from shapely.geometry import Point
from shapely.geometry.polygon import orient


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def box_mask(shape,origin,resolution,objects):
    y,x=np.indices(shape);x=origin[0]+(x+.5)*resolution;y=origin[1]+(y+.5)*resolution
    occupied=np.zeros(shape,bool)
    for obj in objects:
        angle=math.radians(obj.get('yaw_degrees',0.));c,s=math.cos(angle),math.sin(angle)
        dx=x-obj['position'][0];dy=y-obj['position'][1];width,depth=obj['dimensions'][:2]
        occupied |= (np.abs(c*dx+s*dy)<=width/2)&(np.abs(-s*dx+c*dy)<=depth/2)
    return occupied


def scene_clutter(support,free):
    count=int(support.sum())
    if not count:raise ValueError('Empty fixed scene support')
    blocked=int((support&~free).sum())
    return {'support_cells':count,'blocked_cells':blocked,'fraction':blocked/count}


def grid_graph(free,resolution):
    ids=np.full(free.shape,-1,np.int32);ids[free]=np.arange(int(free.sum()),dtype=np.int32)
    rows=[];cols=[];weights=[];h,w=free.shape
    for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
        r0=max(0,-dy);r1=min(h,h-dy);c0=max(0,-dx);c1=min(w,w-dx)
        rr,cc=np.mgrid[r0:r1,c0:c1]
        keep=free[rr,cc]&free[rr+dy,cc+dx]
        if dy and dx:keep &= free[rr+dy,cc]&free[rr,cc+dx]
        a=ids[rr[keep],cc[keep]];b=ids[rr[keep]+dy,cc[keep]+dx]
        rows.extend([a,b]);cols.extend([b,a]);weights.extend([np.full(len(a),resolution*math.hypot(dx,dy))]*2)
    n=int(free.sum())
    graph=sparse.coo_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(cols))),shape=(n,n)).tocsr()
    graph.sort_indices()
    return ids,graph


def free_polygons(free,origin,resolution):
    starts=[];ends=[];ys=[]
    for row,line in enumerate(free):
        d=np.diff(np.r_[False,line,False].astype(np.int8))
        left=np.flatnonzero(d==1);right=np.flatnonzero(d==-1)
        starts.extend(left);ends.extend(right);ys.extend([row]*len(left))
    left=np.asarray(starts);right=np.asarray(ends);y=np.asarray(ys)
    rects=shapely.box(origin[0]+left*resolution,origin[1]+y*resolution,
        origin[0]+right*resolution,origin[1]+(y+1)*resolution)
    geometry=shapely.union_all(rects)
    if geometry.geom_type=='Polygon':return [orient(geometry,sign=1.)]
    return [orient(p,sign=1.) for p in geometry.geoms if p.geom_type=='Polygon']


def reflex_vertices(poly):
    result=[]
    for ring in [poly.exterior,*poly.interiors]:
        points=np.asarray(ring.coords)[:-1]
        incoming=points-np.roll(points,1,axis=0);outgoing=np.roll(points,-1,axis=0)-points
        cross=incoming[:,0]*outgoing[:,1]-incoming[:,1]*outgoing[:,0]
        result.extend(points[cross < -1e-12])
    return np.asarray(result,float).reshape(-1,2)


def polygon_shortest_path(poly,start,goal,upper_bound=math.inf):
    start=np.asarray(start,float);goal=np.asarray(goal,float)
    if poly is None or not poly.covers(Point(start)) or not poly.covers(Point(goal)):return None
    poly=orient(poly,sign=1.)
    # Nanometre predicate tolerance only; it cannot affect qualification margins.
    covered=poly.buffer(1e-9,join_style='mitre');shapely.prepare(covered)
    direct=shapely.LineString([start,goal])
    if covered.covers(direct):return {'length_m':float(np.linalg.norm(goal-start)),'path':[start.tolist(),goal.tolist()],'vertices':0}
    vertices=reflex_vertices(poly)
    vertices=vertices[np.linalg.norm(vertices-start,axis=1)+np.linalg.norm(vertices-goal,axis=1)<=upper_bound+1e-7]
    nodes=np.vstack([start,goal,vertices]);rows=[];cols=[];costs=[]
    for i in range(len(nodes)-1):
        j=np.arange(i+1,len(nodes));length=np.linalg.norm(nodes[j]-nodes[i],axis=1)
        possible=(np.linalg.norm(nodes[i]-start)+length+np.linalg.norm(nodes[j]-goal,axis=1)<=upper_bound+1e-7)
        possible |= (np.linalg.norm(nodes[j]-start,axis=1)+length+np.linalg.norm(nodes[i]-goal)<=upper_bound+1e-7)
        j=j[possible];length=length[possible]
        if not len(j):continue
        lines=shapely.linestrings(np.stack([np.broadcast_to(nodes[i],(len(j),2)),nodes[j]],axis=1))
        valid=shapely.covers(covered,lines);j=j[valid];length=length[valid]
        rows.extend([np.full(len(j),i),j]);cols.extend([j,np.full(len(j),i)]);costs.extend([length,length])
    if not rows:return None
    graph=sparse.coo_matrix((np.concatenate(costs),(np.concatenate(rows),np.concatenate(cols))),shape=(len(nodes),len(nodes))).tocsr()
    distance,pred=dijkstra(graph,directed=False,indices=0,return_predecessors=True)
    if not np.isfinite(distance[1]):return None
    path=[1]
    while path[-1]!=0:
        parent=int(pred[path[-1]])
        if parent<0:raise RuntimeError('Broken visibility predecessor')
        path.append(parent)
    return {'length_m':float(distance[1]),'path':nodes[path[::-1]].tolist(),'vertices':len(vertices)}


def component_polygon(polygons,start,goal):
    a=Point(start);b=Point(goal)
    return next((p for p in polygons if p.covers(a) and p.covers(b)),None)


def load_scene(inputs):
    profile=json.loads((inputs/'scene_profile.json').read_text());objects=json.loads((inputs/'layout.json').read_text())['objects']
    limits=[profile['mapping']['x_limits'],profile['mapping']['y_limits']]
    with np.load(inputs/'navigation_grid.npz') as f:data={k:f[k].copy() for k in f.files}
    resolution=(limits[0][1]-limits[0][0])/(data['free'].shape[1]-1)
    assert abs(resolution-.025)<1e-8
    h=int(round((limits[1][1]-limits[1][0])/resolution));w=int(round((limits[0][1]-limits[0][0])/resolution))
    # The legacy geometry grid has an extra boundary row/column, not extra area.
    support=np.isfinite(data['ground_height'][:h,:w]);origin=[limits[0][0],limits[1][0]]
    boxes=box_mask((h,w),origin,resolution,objects)
    free=data['free'][:h,:w]&support&~boxes
    clearance=np.maximum(0.,ndimage.distance_transform_edt(np.pad(free,1))[1:-1,1:-1]*resolution-resolution/2)
    upper_free=data['box_free'][:h,:w]&support&~boxes
    upper_clearance=np.maximum(0.,ndimage.distance_transform_edt(np.pad(upper_free,1))[1:-1,1:-1]*resolution-resolution/2)
    return profile,origin,resolution,support,free,clearance,boxes,upper_clearance


def qualify(inputs,output):
    output.mkdir(parents=True,exist_ok=True)
    profile,origin,res,support,free,clearance,boxes,upper_clearance=load_scene(inputs)
    clutter=scene_clutter(support,free);clutter.update(total_area_m2=clutter['support_cells']*res*res,blocked_area_m2=clutter['blocked_cells']*res*res,
        scope='Fixed original ground-mesh support; exterior void excluded; low-height collision projections plus all initial boxes; no robot-radius inflation.')
    robot_free=clearance>.30;robot_labels,_=ndimage.label(robot_free)
    raw_ids,raw_graph=grid_graph(free,res);robot_ids,robot_graph=grid_graph(robot_free,res)
    polygons=free_polygons(free,origin,res)
    def cell(xy):return (int((xy[1]-origin[1])/res),int((xy[0]-origin[0])/res))
    def xy(rc):return [origin[0]+(int(rc[1])+.5)*res,origin[1]+(int(rc[0])+.5)*res]
    def length_for(graph,ids,start,goal):
        a,b=cell(start),cell(goal)
        if not(0<=a[0]<ids.shape[0] and 0<=a[1]<ids.shape[1] and 0<=b[0]<ids.shape[0] and 0<=b[1]<ids.shape[1]):return None
        if ids[a]<0 or ids[b]<0:return None
        length=float(dijkstra(graph,indices=int(ids[a]),min_only=True)[int(ids[b])])
        if not np.isfinite(length):return None
        return length+float(np.linalg.norm(np.asarray(start)-xy(a))+np.linalg.norm(np.asarray(goal)-xy(b)))
    original=[]
    for task in json.loads((inputs/'tasks_preflight.json').read_text())['tasks']:
        start=task['start'];goal=task['goal'];euclidean=float(np.linalg.norm(np.asarray(start)-goal))
        upper=length_for(raw_graph,raw_ids,start,goal);robot=length_for(robot_graph,robot_ids,start,goal)
        shortest=None if upper is None else polygon_shortest_path(component_polygon(polygons,start,goal),start,goal,upper)
        ratio=None if shortest is None else shortest['length_m']/euclidean
        row={'id':task['id'],'start':start,'goal':goal,'euclidean_m':euclidean,'shortest_geometry':shortest,
            'tortuosity':ratio,'robot_clearance_grid_path_m':robot,'clutter':clutter['fraction'],
            'qualified':ratio is not None and ratio>1.3 and robot is not None and clutter['fraction']>.2}
        original.append(row);print(json.dumps({'original':row['id'],'tortuosity':ratio,'robot_path_m':robot,'qualified':row['qualified']}),flush=True)
    sizes=np.bincount(robot_labels.ravel());sizes[0]=0;largest=int(np.argmax(sizes))
    safe=(clearance>=.60)&(upper_clearance>=.20)&(robot_labels==largest)
    rows,cols=np.nonzero(safe);indices=np.arange(len(rows));rng=np.random.default_rng(218);rng.shuffle(indices)
    candidates=[]
    for idx in indices:
        rc=(int(rows[idx]),int(cols[idx]));point=np.asarray(xy(rc))
        if all(np.linalg.norm(point-other['xy'])>=.45 for other in candidates):candidates.append({'rc':rc,'xy':point})
        if len(candidates)>=140:break
    if len(candidates)<6:raise RuntimeError('Insufficient safe diverse endpoints')
    source_ids=np.array([raw_ids[c['rc']] for c in candidates],np.int32)
    distances=dijkstra(raw_graph,indices=source_ids)
    pairs=[]
    for i in range(len(candidates)):
        for j in range(i+1,len(candidates)):
            a=candidates[i];b=candidates[j];euclidean=float(np.linalg.norm(a['xy']-b['xy']));upper=float(distances[i,raw_ids[b['rc']]])
            if not(3.0<=euclidean<=10.0 and np.isfinite(upper) and 5.0<=upper<=16.0 and 1.43<=upper/euclidean<=3.0):continue
            score=abs(upper/euclidean-1.8)+.025*abs(upper-9.)
            pairs.append((score,i,j,euclidean,upper))
    del distances
    pairs.sort();selected=[];evaluated=[]
    while pairs and len(selected)<5:
        def choice_score(row):
            score,i,j,_,_=row;a=candidates[i]['xy'];b=candidates[j]['xy']
            if selected:
                diversity=min(min(np.linalg.norm(a-x['start'])+np.linalg.norm(b-x['goal']),
                    np.linalg.norm(a-x['goal'])+np.linalg.norm(b-x['start'])) for x in selected)
                score+=2./max(diversity,.1)
            return score
        item=min(pairs,key=choice_score);pairs.remove(item);_,i,j,euclidean,upper=item
        start=candidates[i]['xy'];goal=candidates[j]['xy']
        directions=[(start,goal),(goal,start)]
        directions=[(a,b) for a,b in directions if not any(np.linalg.norm(a-x['start'])<.45 or np.linalg.norm(b-x['goal'])<.45 for x in selected)]
        if not directions:continue
        initial_heading=[0.,math.pi,math.pi/2,0.,math.pi][len(selected)]
        start,goal=min(directions,key=lambda ab:abs((math.atan2(*(ab[1]-ab[0])[::-1])-initial_heading+math.pi)%(2*math.pi)-math.pi))
        if any(min(np.linalg.norm(start-x['start'])+np.linalg.norm(goal-x['goal']),np.linalg.norm(start-x['goal'])+np.linalg.norm(goal-x['start']))<1.25 for x in selected):continue
        shortest=polygon_shortest_path(component_polygon(polygons,start,goal),start,goal,upper)
        if shortest is None:continue
        ratio=shortest['length_m']/euclidean
        row={'id':f'T{len(selected)+1:02d}','start':start.tolist(),'goal':goal.tolist(),'euclidean_m':euclidean,
            'shortest_geometry':shortest,'tortuosity':ratio,'grid_upper_bound_m':upper,
            'robot_clearance_grid_path_m':length_for(robot_graph,robot_ids,start,goal),'clutter':clutter['fraction'],
            'start_clearance_m':float(clearance[cell(start)]),'goal_clearance_m':float(clearance[cell(goal)]),
            'start_upper_fixture_clearance_m':float(upper_clearance[cell(start)]),'goal_upper_fixture_clearance_m':float(upper_clearance[cell(goal)])}
        row['qualified']=ratio>1.3 and row['robot_clearance_grid_path_m'] is not None and clutter['fraction']>.2
        evaluated.append(row)
        if row['qualified'] and ratio>=1.40:
            selected.append(row);print(json.dumps({'selected':row}),flush=True)
    mesh_audit=inputs/'mesh_geometry_audit.json'
    assert mesh_audit.exists(), 'Whole-instance AABB grids are not accepted for task qualification'
    geometry=json.loads(mesh_audit.read_text())
    report={'assessment_only':True,'navigation_success_not_yet_tested':True,'scene':'MV7','scene_scale':[1,1,1],
        'definition':{'tortuosity':'Continuous shortest geometric path / Euclidean endpoint distance on the fixed raster-derived collision polygon; uninflated geometry is a lower bound for finite-size motion on this occupancy representation.',
            'clutter':'Nontraversable low-height collision footprint / fixed ground-mesh support area, before interaction; exterior void excluded.',
            'disconnected':'Not qualified; infinity is not accepted as a tortuosity score.','resolution_m':res,'predicate_tolerance_m':1e-9,
            'robot_connectivity_screen_radius_m':.30,'endpoint_clearance_m':.60,'online_runtime_map':'Must start empty; assessment map/path must not enter navigator.'},
        'endpoint_visibility_screen':'Also require 0.20m clearance from the existing higher-obstacle layer so endpoints are not under furniture.',
        'software':{'shapely':shapely.__version__,'numpy':np.__version__,'audit_tool_sha256':sha(__file__)},
        'inputs':{name:sha(inputs/name) for name in ['navigation_grid.npz','scene_profile.json','layout.json','tasks_preflight.json','mesh_geometry_audit.json']},
        'geometry_method':geometry['method'],'geometry_mesh_count':geometry['mesh_count'],'geometry_asset_sha256':geometry['asset_sha256'],
        'clutter':clutter,'original_tasks':original,'qualified_candidates':selected,'candidate_endpoint_count':len(candidates),
        'all_five_qualified':len(selected)==5 and all(x['qualified'] for x in selected)}
    (output/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    with np.load(inputs/'navigation_grid.npz') as grid:static_occupied=support&~grid['free'][:support.shape[0],:support.shape[1]]
    np.savez_compressed(output/'evaluation_map.npz',support=support,free=free,static_occupied=static_occupied,robot_free=robot_free,clearance=clearance,boxes=boxes,origin=origin,resolution=res)
    assert report['all_five_qualified']
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=qualify(a.inputs,a.output);print(json.dumps({'clutter':report['clutter'],'all_five_qualified':report['all_five_qualified']}))
