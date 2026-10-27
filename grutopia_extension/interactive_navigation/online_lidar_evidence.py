"""Ray evidence for unknown-map navigation, including measured empty rays.

No-hit range endpoints prove space only up to a conservative sensor range.
Floor/ceiling returns prove only the part of the ray inside the robot's
obstacle-height band. A closer in-band hit clips every ray in its azimuth bin.
"""
import numpy as np
from grutopia_extension.interactive_navigation.mapping import world_ray_cells


def online_lidar_masks(config, origin, points, max_range, min_range=.25):
    shape=(int(np.ceil((config.y_limits[1]-config.y_limits[0])/config.grid_resolution)),
           int(np.ceil((config.x_limits[1]-config.x_limits[0])/config.grid_resolution)))
    hit=np.zeros(shape,bool);free=np.zeros(shape,bool)
    points=np.asarray(points,float).reshape(-1,3);origin=np.asarray(origin,float)
    delta=points-origin
    distance=np.linalg.norm(delta,axis=1);horizontal=np.linalg.norm(delta[:,:2],axis=1)
    # PhysX's no-hit point can be at max_range + min_range. A finite range
    # endpoint still proves traversal to .99*max_range; never mark it occupied.
    valid=(np.isfinite(points).all(axis=1)&(distance>=min_range)&
           (distance<=max_range+min_range+.01)&(horizontal>1e-8))
    delta=delta[valid];points=points[valid];distance=distance[valid];horizontal=horizontal[valid]
    if not len(points):return hit,free,{'rays':0,'range_rays':0,'free_segments':0}
    bins=min(config.lidar_scan_bins,config.max_lidar_rays)
    azimuth=((np.arctan2(delta[:,1],delta[:,0])+np.pi)/(2*np.pi)*bins).astype(int)%bins
    lower,upper=config.obstacle_height
    measured_hit=(distance<max_range*.99)&(points[:,2]>=lower)&(points[:,2]<=upper)
    groups={}
    for i,k in enumerate(azimuth):groups.setdefault(int(k),[]).append(i)
    segments=0;range_rays=0
    def mark(mask,cells):
        for r,c in cells:
            if 0<=r<shape[0] and 0<=c<shape[1]:mask[r,c]=True
    for indices in groups.values():
        obstacles=[i for i in indices if measured_hit[i]]
        nearest=min(obstacles,key=lambda i:horizontal[i]) if obstacles else None
        limit=horizontal[nearest] if nearest is not None else float('inf')
        if nearest is not None:
            cells=world_ray_cells(origin,points[nearest],config)
            if cells:mark(hit,[cells[-1]])
        candidates=[]
        for i in indices:
            end=min(1.,max_range*.99/distance[i],limit/horizontal[i])
            dz=delta[i,2]
            if abs(dz)<1e-9:
                if not lower<=origin[2]<=upper:continue
                first,last=0.,end
            else:
                t0,t1=sorted(((lower-origin[2])/dz,(upper-origin[2])/dz))
                first,last=max(0.,t0),min(end,t1)
            if last-first<1e-8:continue
            candidates.append((i,first,last,horizontal[i]*first,horizontal[i]*last))
        if not candidates:continue
        # Two real beams per azimuth retain near-floor and long-range
        # coverage without turning a dense vertical fan into synthetic rays.
        chosen={max(candidates,key=lambda x:x[4]),min(candidates,key=lambda x:(x[3],-x[4]))}
        for i,first,last,_,_ in chosen:
            a=origin+first*delta[i];b=origin+last*delta[i]
            mark(free,world_ray_cells(a,b,config))
            segments+=1;range_rays+=int(distance[i]>=max_range*.99)
    free &= ~hit
    return hit,free,{'rays':len(points),'range_rays':range_rays,'free_segments':segments}
