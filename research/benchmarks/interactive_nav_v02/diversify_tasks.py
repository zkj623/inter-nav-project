"""Resample tasks in fixed scenes by endpoint coverage, length strata, and route overlap."""
import argparse,json,itertools,math,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse.csgraph import dijkstra
from upgrade_connectivity import graph,route,asset_kernels,nav_with,R,STRUCT,dump

def corridor(cells):
    """0.25 m route cells with a one-cell Chebyshev buffer, direction independent."""
    q=np.unique(np.asarray(cells)//10,axis=0)
    return frozenset((int(y+dy),int(x+dx)) for y,x in q for dy in (-1,0,1) for dx in (-1,0,1))

def overlap(a,b):
    n=len(a&b)
    return n/(len(a)+len(b)-n)

MIN_ROUTE_M=3.0

def sample(d,gdata,kernels,reserve_controls=False):
    items=d['objects'];origin=gdata['origin'];nominal=gdata['nominal'];initial=gdata['initial_nominal'];final=gdata['cleared_nominal'];opt=gdata['initial_optimistic']
    initial_labels,_=ndi.label(opt,STRUCT);clear=ndi.distance_transform_edt(~gdata['blocked'])*R
    g,cells,index=graph(final);safe=initial&final&(clear>.40);eligible=np.flatnonzero(safe[tuple(cells.T)])
    landmarks=[int(eligible[0])];minimum=np.full(len(cells),np.inf);distances=[];preds=[]
    for _ in range(80):
        dd,pp=dijkstra(g,indices=landmarks[-1],return_predecessors=True);distances.append(dd);preds.append(pp);minimum=np.minimum(minimum,dd)
        remaining=np.where(np.isfinite(minimum[eligible]),minimum[eligible],-1);nxt=int(eligible[np.argmax(remaining)])
        if minimum[nxt]<.7:break
        landmarks.append(nxt)
    landmarks=landmarks[:len(distances)];distances=np.array(distances);preds=np.array(preds)
    # The early farthest-point samples form spatial zones separated by >= 2 m geodesic distance.
    centers=[0]
    for j in range(1,len(landmarks)):
        if min(distances[k,landmarks[j]] for k in centers)>=2 and len(centers)<10:centers.append(j)
    zones=np.argmin(distances[centers][:,landmarks],axis=0)
    zone_meta=[{'id':f'R{k+1:02}','center_xy_m':(origin+cells[landmarks[j]][::-1]*R).tolist()} for k,j in enumerate(centers)]
    subsets=[]
    for n in range(1,len(items)+1):
        for ss in itertools.combinations(range(len(items)),n):
            labels,_=ndi.label(nav_with(gdata['optimistic'],items,kernels,'optimistic',ss),STRUCT);subsets.append((n,list(ss),labels))
    goal_gap={int(label):ndi.distance_transform_edt(initial_labels!=label)*R for label in np.unique(initial_labels[safe]) if label}
    candidates=[]
    for i,j in itertools.combinations(range(len(landmarks)),2):
        s,t=landmarks[i],landmarks[j];sc,tc=tuple(cells[s]),tuple(cells[t]);ls,lt=int(initial_labels[sc]),int(initial_labels[tc]);length=float(distances[i,t])
        if not np.isfinite(length) or length<MIN_ROUTE_M or not ls or not lt or ls==lt:continue
        if goal_gap[ls][tc]<=.5 or goal_gap[lt][sc]<=.5:continue
        solutions=[(n,ss) for n,ss,ll in subsets if ll[sc] and ll[sc]==ll[tc]]
        if not solutions:continue
        n=solutions[0][0];rr=route(cells,preds[i],s,t)
        candidates.append({'i':i,'j':j,'start':sc,'goal':tc,'length':length,'cells':rr,'corridor':corridor(rr),'zone_pair':tuple(sorted((int(zones[i]),int(zones[j])))),'solutions':[ss for nn,ss in solutions if nn==n],'minimum':n,'class':'interaction_required'})
    assert candidates,d['id']
    longest=max(c['length'] for c in candidates);short=max(MIN_ROUTE_M+.75,.35*longest);medium=max(short+.5,.7*longest)
    def category(length):return 'short' if length<=short else 'medium' if length<=medium else 'long'
    for c in candidates:c['length_band']=category(c['length'])
    selected=[];covered_zones=set();zone_counts=Counter();used_pairs=Counter();used_points=Counter();occupied=set();covered_objects=set();bins=Counter()
    # Quotas prevent longest routes from dominating; spatial novelty is evaluated before length.
    target=Counter(short=3,medium=3,long=2)
    def choose(pool,required=True,flexible_quota=False):
        eligible=[]
        for c in pool:
            if required and bins[c['length_band']] >= (4 if flexible_quota else target[c['length_band']]):continue
            if used_pairs[c['zone_pair']]>=max(2,math.ceil(8/max(1,len(centers)))):continue
            if used_points[c['start']]>=2 or used_points[c['goal']]>=2:continue
            ov=max((overlap(c['corridor'],s['corridor']) for s in selected),default=0.)
            if ov>.65:continue
            zs=set(c['zone_pair']);new_zones=len(zs-covered_zones);novel=len(c['corridor']-occupied)/len(c['corridor'])
            end_spread=min((min(np.linalg.norm(np.array(c['start'])-np.array(s[k]))*R,np.linalg.norm(np.array(c['goal'])-np.array(s[k]))*R) for s in selected for k in ['start','goal']),default=2.)
            new_objects=max((len(set(ss)-covered_objects) for ss in c.get('solutions',[[]])),default=0)
            balance=sum(1/(1+zone_counts[z]) for z in c['zone_pair'])
            score=6*new_zones+4*balance+2*(used_pairs[c['zone_pair']]==0)+2*novel-3*ov+.5*min(end_spread,2)+.25*new_objects-.4*(used_points[c['start']]+used_points[c['goal']])
            eligible.append((score,c))
        if not eligible:return None
        return max(eligible,key=lambda q:(q[0],-abs(q[1]['length']-{'short':short*.7,'medium':(short+medium)/2,'long':(medium+longest)/2}[q[1]['length_band']])))[1]
    def take(c):
        selected.append(c);covered_zones.update(c['zone_pair']);zone_counts.update(c['zone_pair']);used_pairs[c['zone_pair']]+=1;used_points[c['start']]+=1;used_points[c['goal']]+=1;occupied.update(c['corridor'])
        if c['class']=='interaction_required':bins[c['length_band']]+=1
        if c.get('solutions'):
            c['witness']=max(c['solutions'],key=lambda ss:len(set(ss)-covered_objects));covered_objects.update(c['witness'])
    # Select controls from the same distributed landmarks, using shortest paths with objects absent.
    bg,bc,bi=graph(nominal);controls=[]
    for i,s in enumerate(landmarks):
        sc=tuple(cells[s]);dd,pp=dijkstra(bg,indices=int(bi[sc]),return_predecessors=True)
        for j in range(i+1,len(landmarks)):
            tc=tuple(cells[landmarks[j]]);t=int(bi[tc]);length=float(dd[t])
            if not np.isfinite(length) or length<MIN_ROUTE_M:continue
            rr=route(bc,pp,int(bi[sc]),t)
            if not initial[tuple(rr.T)].all():continue
            controls.append({'i':i,'j':j,'start':sc,'goal':tc,'length':length,'cells':rr,'corridor':corridor(rr),'zone_pair':tuple(sorted((int(zones[i]),int(zones[j])))),'minimum':0,'class':'no_interaction_optimal','length_band':category(length)})
    if reserve_controls:
        for _ in range(2):
            c=choose(controls,False)
            if c is None:raise RuntimeError(d['id']+': fewer than two distinct controls even before required selection')
            take(c);controls.remove(c)
    for _ in range(8):
        c=choose(candidates)
        if c is None:c=choose(candidates,flexible_quota=True)
        if c is None:raise RuntimeError(f"{d['id']}: insufficient distinct required tasks; reserved_controls={reserve_controls}, selected_bins={dict(bins)}")
        take(c);candidates.remove(c)
    assert all(bins[b]>=1 for b in ('short','medium','long')),d['id']+' missing a length stratum'
    if not reserve_controls:
        for _ in range(2):
            c=choose(controls,False)
            if c is None:return sample(d,gdata,kernels,reserve_controls=True)
            take(c);controls.remove(c)
    selected.sort(key=lambda c:c['class']=='no_interaction_optimal')
    tasks=[];start_counts=Counter();goal_counts=Counter()
    for k,c in enumerate(selected):
        iz,jz=int(zones[c['i']]),int(zones[c['j']]);rr=c['cells']
        if start_counts[iz]+goal_counts[jz]>start_counts[jz]+goal_counts[iz]:rr=rr[::-1];iz,jz=jz,iz
        start_counts[iz]+=1;goal_counts[jz]+=1;xy=origin+rr[:,::-1]*R
        t={'id':f"{d['id']}-T{k+1:02}",'class':c['class'],'initial_optimistic_connected':c['class']=='no_interaction_optimal','minimum_relocations_in_optimistic_model':c['minimum'],'start_cell':rr[0].tolist(),'goal_cell':rr[-1].tolist(),'start_position_m':[*xy[0].tolist(),0.],'goal_position_m':[*xy[-1].tolist(),0.],'start_yaw_rad':float(math.atan2(*(xy[1]-xy[0])[::-1])),'reference_route_xy_m':xy.tolist(),'cleared_route_length_m':c['length'],'goal_tolerance_m':.5,'validation':'G2 planar navigation model; robot manipulation rollout pending','length_band':c['length_band'],'start_region':f'R{iz+1:02}','goal_region':f'R{jz+1:02}'}
        if c['class']=='interaction_required':t['optimistic_relocation_subset']=c['witness']
        else:t['optimality']='shortest route with added objects absent is feasible with all added objects in initial poses'
        tasks.append(t)
    overlaps=[overlap(x['corridor'],y['corridor']) for x,y in itertools.combinations(selected,2)]
    before=[corridor(np.rint((np.array(t['reference_route_xy_m'])-origin)[:,::-1]/R).astype(int)) for t in d['tasks']]
    old_overlaps=[overlap(x,y) for x,y in itertools.combinations(before,2)]
    report={'scene':d['id'],'endpoint_regions_total':len(centers),'endpoint_regions_used':len(covered_zones),'distinct_region_pairs':len(used_pairs),'unique_endpoint_positions':len(used_points),'maximum_endpoint_reuse':max(used_points.values()),'length_band_counts':dict(Counter(t['length_band'] for t in tasks)),'required_length_band_counts':dict(bins- Counter(t['length_band'] for t in tasks[8:])),'length_band_thresholds_m':{'short_max':short,'medium_max':medium},'route_corridor_iou_max_before':max(old_overlaps),'route_corridor_iou_max_after':max(overlaps),'route_corridor_iou_mean_before':float(np.mean(old_overlaps)),'route_corridor_iou_mean_after':float(np.mean(overlaps)),'route_lengths_m':[round(t['cleared_route_length_m'],3) for t in tasks],'layout_unchanged':True,'corridor_definition':'0.25 m cells, one-cell Chebyshev dilation; direction-independent Jaccard overlap'}
    report['required_length_band_counts']=dict(bins)
    report['minimum_reference_route_m']=MIN_ROUTE_M
    d['tasks']=tasks;d['endpoint_regions']=zone_meta;d['task_diversity']=report;d['long_route_sampling']='spatial endpoint coverage, short/medium/long strata, route-corridor overlap <= 0.65';return d,report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--objects',type=Path,required=True);ap.add_argument('--only');a=ap.parse_args();kernels={};results=[];updates=[]
    for ep in sorted(a.out.glob('S*/episodes.json')):
        d=json.loads(ep.read_text())
        if a.only and d['id']!=a.only:continue
        original=json.dumps(d['objects'],sort_keys=True)
        for o in d['objects']:
            key=o['asset_id'],o['yaw_deg']
            if key not in kernels:kernels[key]=asset_kernels(a.objects/(key[0]+'.npz'),key[1])
        new,report=sample(d,np.load(ep.parent/'evaluation_grid.npz'),kernels);assert original==json.dumps(new['objects'],sort_keys=True);updates.append((ep,new));results.append(report);print(json.dumps(report),flush=True)
    # Atomic at the suite level with respect to selection failures: don't publish partial selections.
    for ep,new in updates:dump(ep,new)
    dump(a.out/'task_diversity.json',{'passed':True,'scenes':results,'layout_changes':0,'route_corridor_iou_limit':.65})
if __name__=='__main__':main()
