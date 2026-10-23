"""Conservative visual-and-collision envelopes for the bilateral lift task."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def origin_transform(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        transform[:3, :3] = Rotation.from_euler(
            'xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
        ).as_matrix()
    return transform


class RobotCollisionModel:
    def __init__(self, urdf_path):
        import fcl
        import trimesh

        self.fcl = fcl
        root = ET.parse(urdf_path).getroot()
        self.joints = []
        self.parents = {}
        for joint in root.findall('joint'):
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            axis_element = joint.find('axis')
            axis = np.array((1.0, 0.0, 0.0)) if axis_element is None else np.fromstring(
                axis_element.get('xyz'), sep=' '
            )
            self.joints.append((joint.get('name'), parent, child, joint.get('type'),
                                origin_transform(joint.find('origin')), axis))
            self.parents[child] = parent
        self.envelopes = {}
        self.meshes = {}
        self.vertices = {}
        self.faces = {}
        self.visual_vertices = {}
        self.proxy_bounds = {}
        for link in root.findall('link'):
            name = link.get('name')
            geometries = []
            visual_geometries = []
            proxy_geometries = []
            for kind in ('visual', 'collision'):
                for element in link.findall(kind):
                    geometry = element.find('geometry')
                    mesh_element = geometry.find('mesh')
                    if mesh_element is not None:
                        filename = Path(mesh_element.get('filename'))
                        if not filename.is_absolute():
                            filename = Path(urdf_path).parent / filename
                        mesh = trimesh.load(filename, force='mesh', process=False)
                        mesh.apply_scale(np.fromstring(mesh_element.get('scale', '1 1 1'), sep=' '))
                    elif geometry.find('box') is not None:
                        mesh = trimesh.creation.box(np.fromstring(geometry.find('box').get('size'), sep=' '))
                    elif geometry.find('cylinder') is not None:
                        cylinder = geometry.find('cylinder')
                        mesh = trimesh.creation.cylinder(float(cylinder.get('radius')),
                                                         float(cylinder.get('length')), sections=32)
                    else:
                        raise ValueError(f'Unsupported geometry on {name}')
                    mesh.apply_transform(origin_transform(element.find('origin')))
                    geometries.append(mesh)
                    (visual_geometries if kind == 'visual' else proxy_geometries).append(mesh)
            if not geometries:
                continue
            combined = trimesh.util.concatenate(geometries)
            hull = combined.convex_hull
            polygons = np.column_stack((np.full(len(hull.faces), 3), hull.faces)).astype(np.int32).ravel()
            self.envelopes[name] = fcl.Convex(np.asarray(hull.vertices), len(hull.faces), polygons)
            self.vertices[name] = np.asarray(hull.vertices)
            self.faces[name] = np.asarray(hull.faces, dtype=np.int32)
            visual = trimesh.util.concatenate(visual_geometries) if visual_geometries else combined
            visual_hull = visual.convex_hull
            self.visual_vertices[name] = np.asarray(visual_hull.vertices)
            model = fcl.BVHModel()
            model.beginModel(len(visual.vertices), len(visual.faces))
            model.addSubModel(np.asarray(visual.vertices), np.asarray(visual.faces, dtype=np.int32))
            model.endModel()
            self.meshes[name] = model
            if proxy_geometries:
                self.proxy_bounds[name] = trimesh.util.concatenate(proxy_geometries).bounds.tolist()
        self.names = list(self.envelopes)
        self.noncontact_names = [name for name in self.names if 'gripper_finger' not in name]
        self.self_pairs = []
        for index, first in enumerate(self.names):
            for second in self.names[index + 1:]:
                first_family = {first, self.parents.get(first)}
                second_family = {second, self.parents.get(second)}
                if first_family & second_family:
                    continue
                self.self_pairs.append((first, second))

    def forward(self, positions, base=None):
        transforms = {'base_link': np.eye(4) if base is None else np.asarray(base)}
        pending = self.joints.copy()
        while pending:
            next_pending = []
            for name, parent, child, kind, origin, axis in pending:
                if parent not in transforms:
                    next_pending.append((name, parent, child, kind, origin, axis))
                    continue
                motion = np.eye(4)
                value = positions.get(name, 0.0)
                if kind in ('revolute', 'continuous'):
                    motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
                elif kind == 'prismatic':
                    motion[:3, 3] = axis * value
                transforms[child] = transforms[parent] @ origin @ motion
            if len(next_pending) == len(pending):
                raise ValueError('URDF joints do not form a connected tree')
            pending = next_pending
        return transforms

    def objects(self, transforms, exact=False):
        geometries = self.meshes if exact else self.envelopes
        return {name: self.fcl.CollisionObject(geometry, self.fcl.Transform(
            transforms[name][:3, :3], transforms[name][:3, 3]
        )) for name, geometry in geometries.items()}

    def box_clearances(self, transforms, box_transform, dimensions, include_fingers=False):
        obstacle = self.fcl.CollisionObject(self.fcl.Box(*dimensions), self.fcl.Transform(
            box_transform[:3, :3], box_transform[:3, 3]
        ))
        request = self.fcl.DistanceRequest(enable_signed_distance=True)
        objects = self.objects(transforms)
        names = self.names if include_fingers else self.noncontact_names
        return {name: float(self.fcl.distance(objects[name], obstacle, request, self.fcl.DistanceResult()))
                for name in names}

    def self_intersections(self, transforms):
        objects = self.objects(transforms)
        exact = None
        bounds = {}
        for name, vertices in self.vertices.items():
            world_vertices = vertices @ transforms[name][:3, :3].T + transforms[name][:3, 3]
            bounds[name] = (world_vertices.min(axis=0), world_vertices.max(axis=0))
        intersections = []
        request = self.fcl.CollisionRequest()
        for first, second in self.self_pairs:
            if np.any(bounds[first][0] > bounds[second][1]) or np.any(bounds[second][0] > bounds[first][1]):
                continue
            if not self.fcl.collide(objects[first], objects[second], request, self.fcl.CollisionResult()):
                continue
            if exact is None:
                exact = self.objects(transforms, exact=True)
            if self.fcl.collide(exact[first], exact[second], request, self.fcl.CollisionResult()):
                intersections.append([first, second])
        return intersections

    def proxy_coverage_report(self):
        return {name: {'visual_bounds_m': [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
                       'original_collision_bounds_m': self.proxy_bounds.get(name)}
                for name, vertices in self.visual_vertices.items()}
