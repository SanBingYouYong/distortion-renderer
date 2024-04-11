from pprint import pprint
from typing import List, Tuple, Iterable
from enum import Enum
import bpy
from bpy.types import Object
import bmesh
from bmesh.types import BMesh, BMEdge, BMVert
import random
import os
from mathutils import Vector, Matrix, Euler
from math import radians, degrees, atan2, pi
# try: 
#     from dataset_creation.gs_based.gs_grammar import Component
#     from dataset_creation.gs_based.gs_io_handler import IOHandler
# except:
#     try:
#         from gs_grammar import Component
#         from gs_io_handler import IOHandler
#     except:
#         pass


class DRUtils():
    '''
    Utility methods for Dedicated Renderers. 
    '''
    # check gs_grammar.Interpreter.node_tree_names
    straight_comps = [
        (0, 0), (0, 2), 
        (1, 0), (1, 1), (1, 2), 
        (2, 0), (2, 1), (2, 2), 
        (3, 0), (3, 1), (3,2)
    ]
    cylindrical_comps = [(0, 1), (0, 3)]
    sphered_comps = [(1, 3)]

    class DeRegLevels(Enum):
        PERFECT = 0
        LIGHT = 0.01
        MEDIUM = 0.025
        HEAVY = 0.05

    # @staticmethod
    # def read_dataset(dataset_name: str, paramcsv_name: str=None, override_path: str=None):
    #     '''
    #     Read in parameters from target dataset. Uses GS.IOHandler. 
    #     '''
    #     _, img_folder, params = IOHandler.dataset_param_ops(dataset_name, mode=IOHandler.DatasetOps.Read, dataset_paramcsv_name=paramcsv_name, override_path=override_path)
    #     return img_folder, params
    
    @staticmethod
    # def spawn(type: str, subtype: str, params: list, obj_name: str=None) -> Tuple[Component, Object]:
    #     '''
    #     Spawn the target component, then convert to mesh (visual geometry to mesh in Ctrl A menu)
    #     Adds rotation to main body after conversion. 
    #     '''
    #     # spawn component
    #     no_rot_params = params[:]
    #     no_rot_params[3:6] = [0, 0, 0]
    #     comp = Component(type=type, subtype=subtype, params=no_rot_params, obj_name=obj_name)
    #     obj = comp.instantiate()
    #     # convert to mesh
    #     bpy.ops.object.convert(target="MESH")
    #     # apply rotation separately on final mesh
    #     rot = [float(val) for val in params[3:6]]
    #     obj.rotation_euler = Euler(rot, 'XYZ')
    #     bpy.ops.object.transform_apply(rotation=True)
    #     return comp, obj
    
    @staticmethod
    def set_up_blender():
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        # primitives = bpy.data.texts["gs_primitives.py"]
        # exec(primitives.as_string())
        # make cam use freestyle
        bpy.data.scenes["Scene"].render.use_freestyle = True
        freestyle_settings = bpy.context.scene.view_layers["ViewLayer"].freestyle_settings
        lineset = freestyle_settings.linesets["LineSet"]
        lineset.select_silhouette = False
        lineset.select_crease = False
        lineset.select_border = False
        lineset.select_edge_mark = True
    
    @staticmethod
    def get_sharps(obj: Object, threshold: float=0.523599) -> Tuple[List[int], List[int], List[int]]:
        '''
        Select sharp edges, return selected verts, edges and faces, not in order. 

        threshold: sharpness: 0.523599 radian <- 30 degrees, normal
        '''
        bpy.ops.object.editmode_toggle()
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bpy.ops.mesh.select_mode(type='EDGE')
        bpy.ops.mesh.select_all(action='DESELECT')
        # select sharp edges
        bpy.ops.mesh.edges_select_sharp(sharpness=threshold)
        obj.update_from_editmode()
        # reference to verts, edges, faces
        selected_verts = [v.index for v in bm.verts if v.select]
        selected_edges = [e.index for e in bm.edges if e.select]
        selected_faces = [f.index for f in bm.faces if f.select]
        bpy.ops.object.editmode_toggle()
        return selected_verts, selected_edges, selected_faces
    
    @staticmethod
    def get_visibles(obj: Object, mode='VERT', bm: BMesh=None) -> Tuple[List[int], List[int], List[int]]:
        '''
        Select visible vertices, return selected elements according to mode (default vert)
        Note: it leaves the visible vert/edge/face selected

        @params
        mode: VERT/EDGE/FACE. EDGE mode is for straight line rendering in deciding better visibilties. 
        bm: BMesh. Leave as default (none) means this method will obtain its own bmesh through toggling editmode. 
            Potentially ruinning other bmesh in script's context. 
            Giving it a bmesh will stop it from doing so: it now operates on the given bmesh directly, no longer
                toggling editmode. 
        '''
        toggled = False
        if not bm:
            bpy.ops.object.editmode_toggle()
            me = obj.data
            bm = bmesh.from_edit_mesh(me)
            toggled = True
        bpy.ops.mesh.select_mode(type=mode)
        bpy.ops.mesh.select_all(action='DESELECT')
        # select visible verts
        DRUtils._select_border(bpy.context)
        obj.update_from_editmode()
        selected_verts = [v.index for v in bm.verts if v.select]
        selected_edges = [e.index for e in bm.edges if e.select]
        selected_faces = [f.index for f in bm.faces if f.select]
        if toggled:
            bpy.ops.object.editmode_toggle()
        return selected_verts, selected_edges, selected_faces
    
    @staticmethod
    def _getView3dAreaAndRegion(context):
        for area in context.screen.areas: 
            if area.type == "VIEW_3D":    
                for region in area.regions:
                    if region.type == "WINDOW":
                        return area, region
        print(f"Cannot find view3d area and region in context: {context}")

    @staticmethod
    def _select_border(context, view3dAreaAndRegion=None, extend=True):
        '''
        Selects visible elements from viewport
        Assumption: in camera view
        Note: Leaves elements selected. 
        '''
        if not view3dAreaAndRegion:
            view3dAreaAndRegion = DRUtils._getView3dAreaAndRegion(context)
        view3dArea, view3dRegion = view3dAreaAndRegion
        override = context.copy()
        override['area'] = view3dArea
        override['region'] = view3dRegion
        bpy.ops.view3d.select_box(override,xmin=0,xmax=view3dArea.width,ymin=0,ymax=view3dArea.height,mode='ADD')
    
    @staticmethod
    def disturb_coordinate(vert: BMVert | Vector, max_disturb: float, min_disturb: float=None, disturb_axis: int=None) -> Vector:
        '''
        Adds random offset (between max and min) on one of the axis of the given vert or Vector. 
        Returns disturbed Vector coord. 
        
        @params
        min_disturb: default not specified -> negative max disturb
        disturb_axis: default not specified -> random axis xyz (0, 1, 2)  TODO: add multiple axis disturbance? 
        '''
        if isinstance(vert, BMVert):
            coord = Vector(vert.co)
        else:
            coord = vert
        if not min_disturb:
            min_disturb = -max_disturb
        offset_amount = random.uniform(min_disturb, max_disturb)
        offset_axis = random.randint(0, 2) if not disturb_axis else disturb_axis
        coord[offset_axis] += offset_amount
        return coord
    
    @staticmethod
    def spawn_bezier_curve_through(points: Iterable[Vector], obj_name: str=None) -> Object:
        '''
        Spawn a bezier curve passing input Vector coordinates. Returns the spawned curve object. 
        '''
        curve = bpy.data.curves.new(name="BezierCurve", type="CURVE")
        curve.dimensions = "3D"
        spline = curve.splines.new("BEZIER")
        spline.bezier_points.add(len(points) - 1)
        for i, point in enumerate(points):
            bezier_point = spline.bezier_points[i]
            bezier_point.co = point
            # type auto will align handles together
            bezier_point.handle_left_type = "AUTO"
            bezier_point.handle_right_type = "AUTO"
        curve_name = f"BezierCurve_{obj_name}_" if obj_name else "BezierCurve_Obj_"
        obj = bpy.data.objects.new(curve_name, curve)
        bpy.context.collection.objects.link(obj)
        return obj
    
    @staticmethod
    def spawn_curves(curves: List[Iterable[Vector]], obj_name: str=None) -> List[Object]:
        spawned_curves = []
        for curve in curves: 
            spawned_curves.append(
                DRUtils.spawn_bezier_curve_through(curve, obj_name=obj_name)
            )
        return spawned_curves
    
    @staticmethod
    def _verts_from_edges(edges: List[int], bm: BMesh) -> List[int]:
        '''
        Extracts non-repeating verts from edges. Returns verts, index. 
        '''
        all_verts = set()
        for edge in edges:
            v0, v1 = bm.edges[edge].verts
            all_verts.add(v0.index)
            all_verts.add(v1.index)
        return list(all_verts)

    @staticmethod
    def _edges_from_verts(verts: List[int], bm: BMesh) -> List[int]:
        '''
        Returns edge index for edges whose both verts are in input verts list. 
        '''
        edges = set()
        for edge in bm.edges:
            v0, v1 = edge.verts
            if v0.index in verts and v1.index in verts:
                edges.add(edge.index)
        return list(edges)
    
    # Half Sphere Pipeline Helpers: 
    @staticmethod
    def _floats_similar(f1: float, f2: float, tolerance: float=0.25):
        return abs(f1 - f2) < tolerance

    @staticmethod
    def _find_diff_edge(edge_angs: List[Tuple[bmesh.types.BMEdge, float]]) -> Tuple[bmesh.types.BMEdge, float]:
        if DRUtils._floats_similar(edge_angs[0][1], edge_angs[1][1]):
            return edge_angs[2]
        elif DRUtils._floats_similar(edge_angs[1][1], edge_angs[2][1]):
            return edge_angs[0]
        elif DRUtils._floats_similar(edge_angs[2][1], edge_angs[0][1]):
            return edge_angs[1]
        else:
            raise ValueError(f"Cannot find the up-pointing edge among: {edge_angs}")

    @staticmethod
    def _sort_by_z(vert: BMVert):
        return vert.co[-1]
    
    # Disturb Coordinates
    @staticmethod
    def _disturb_edge(v0: Vector, v1: Vector, max_disturb_factor=0.05) -> Tuple[Vector, Vector, Vector]:
        '''
        Adds random offset to v0, v1, and their mid points. 
        Returns a 3-tuple of Vector. 
        '''
        mid = Vector(
            [(val0 + val1)/ 2 for val0, val1 in zip(v0, v1)]
        )
        length = mid.length
        max_disturb = max_disturb_factor * length
        points = [
            DRUtils.disturb_coordinate(v0, max_disturb), 
            DRUtils.disturb_coordinate(mid, max_disturb),  # TODO: disturb mid or not? or downgrade mid point disturbance? 
            DRUtils.disturb_coordinate(v1, max_disturb)
        ]
        return points

    @staticmethod
    def disturb_edges(edges: List[Tuple[int, int]], obj: Object, 
                      de_reg: DeRegLevels=DeRegLevels.MEDIUM) -> List[Tuple[Vector, Vector, Vector]]:
        '''
        @params
        edges: [(vert 0 index, vert 1 index)...]. Note that it's not BMEdge or blender edge, but two endpoints representing
        that there should be a line between these two vertices. 
        '''
        disturbed_coords: List[Tuple[Vector, Vector, Vector]] = []
        bpy.ops.object.editmode_toggle()
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        for (v0, v1) in edges:
            vert0, vert1 = bm.verts[v0], bm.verts[v1]
            disturbed_coords.append(
                DRUtils._disturb_edge(vert0.co, vert1.co, max_disturb_factor=de_reg.value)
            )
        bpy.ops.object.editmode_toggle()
        return disturbed_coords
    
    @staticmethod
    def _disturb_verts_on_curve(points: List[Vector], max_disturb_factor=0.05, mid_disturb_downgrade=0.5) -> List[Vector]:
        '''
        Add disturbance to start, end and all points in middle and return
        Disturbance factor: still start to end length. 
        Uses a smaller offset as midpoints common offset
        '''
        disturbed_points = []
        start, end = points[0], points[-1]
        # calculate distance between start and end
        x2 = (start.x - end.x) ** 2
        y2 = (start.y - end.y) ** 2
        z2 = (start.z - end.z) ** 2
        length = (x2+y2+z2)**.5
        max_disturb = length * max_disturb_factor
        # curve verts disturb: start, end, mids
        start = DRUtils.disturb_coordinate(start, max_disturb)
        disturbed_points.append(start)
        # mid: 
        # determine a common offset
        mid_offset = max_disturb * mid_disturb_downgrade
        _offset_amount = random.uniform(-mid_offset, mid_offset)
        _offset_axis = random.randint(0, 2)
        for point in points[1:-1]: 
            disturbed_point = Vector(point.co) if isinstance(point, BMVert) else Vector(point)
            disturbed_point[_offset_axis] += _offset_amount
            disturbed_points.append(disturbed_point)
        # end:
        end = DRUtils.disturb_coordinate(end, max_disturb)
        disturbed_points.append(end)
        return disturbed_points
    
    @staticmethod
    def disturb_curves(curves: List[List[Vector]], obj: Object, 
                       de_reg: DeRegLevels = DeRegLevels.MEDIUM) -> List[List[Vector]]:
        '''
        Takes in curves (curve: ordered list of vert coordinate in Vector), return disturbed versions. 
        '''
        disturbed_curves = []
        for curve in curves:
            disturbed_curves.append(
                DRUtils._disturb_verts_on_curve(curve, de_reg.value)
            )
        return disturbed_curves
    
    @staticmethod
    def find_curve_endpoints(unordered_verts: List[int], unordered_edges: List[int], bm: BMesh) -> Tuple[int, int]:
        '''
        Find the start and end vertices (verts with only one visharp edge)
        Returns (start, end) index
        If it's a circle (all verts has two visharp edges), returns a random vert both as start and as end. 
        '''
        if len(unordered_verts) == 2:  # early termination
            return unordered_verts
        bm.verts.ensure_lookup_table()
        # find start/end (only one edge included in visharp)
        start = None
        end = None
        all_two_visharp_edges = True  # in a circle
        for vert in unordered_verts:
            vert_edges = bm.verts[vert].link_edges
            qualifier = 0
            for vert_edge in vert_edges:
                if vert_edge.index in unordered_edges:
                    qualifier += 1
            if qualifier == 1:
                if not start:
                    start = vert
                elif not end:
                    end = vert
                else:
                    raise AssertionError(f"More than 2 start/end verts found: {vert}")
            if qualifier != 2:
                all_two_visharp_edges = False
            if start and end:
                break
        if all_two_visharp_edges:
            # a circle
            # print(f"Find circle: {unordered_verts} --- {unordered_edges}")
            return unordered_verts[0], unordered_verts[0]
        if not start or not end:
            raise AssertionError(f"Failed to find start or end: {start}, {end}; in : {unordered_verts} --- {unordered_edges}")
        return start, end
    
    @staticmethod
    def get_curve_points(curve_edges: List[int], bm: BMesh) -> Tuple[List[int], List[Vector]]:
        '''
        Find the start and end vertices (verts with only one visharp edge)
        use vert.link_edges to determine the right outgoing edge at each step. 
        Returns list of vertices index and Vector coords, in order. 
        Note: this breaks when the curve is occluded by other geometries. 
        TODO: fix this so that we can freely generalize to any curve.
        '''
        # print(f"received curve edges: {curve_edges}")
        curve_verts = DRUtils._verts_from_edges(curve_edges, bm) # extract verts from edges.verts. index
        # print(f"extracted curve verts: {curve_verts}")
        # determine start and end vertices. 
        start, end = DRUtils.find_curve_endpoints(curve_verts, curve_edges, bm)
        # print(f"start, end: {start}, {end}")
        # original get_bottom_curve
        ordered_verts = [start]
        unordered_verts = curve_verts[:]
        if start == end:  # circle
            unordered_verts.append(end)  # add end again
        unordered_verts.remove(start)
        # print(f"ordered vert ini: {ordered_verts}")
        # print(f"unordered verts ini: {unordered_verts}")
        _i = 0
        _i_max = len(curve_verts) * 2  # safe limit with approx. margin
        bm.verts.ensure_lookup_table()
        while unordered_verts != []:
            _i += 1
            current_contour = ordered_verts[-1]
            # print(f"current contour: {current_contour}")
            found_next = False
            for contour_edge in bm.verts[current_contour].link_edges:
                # if it's a circle, this will decide which way to go randomly
                other_vert = [v for v in contour_edge.verts if v.index != current_contour][0]
                # print(f"testing the other vert: {other_vert}")
                if other_vert.index in unordered_verts:  # find the next vert on the path
                    if len(ordered_verts) == 2 and other_vert.index == start:  # avoid going back again if it's a circle
                        continue  # skip this edge
                    ordered_verts.append(other_vert.index)
                    unordered_verts.remove(other_vert.index)
                    found_next = True
                    # print(f"found next: {other_vert}")
                    # print(f"updated verts: {unordered_verts} --- {ordered_verts}")
                    break
            if not found_next:
                raise AssertionError(f"Merged get-curve-point cannot find next vert on path: {unordered_verts} --- {ordered_verts}")
            if _i >= _i_max:
                raise AssertionError(f"Maximum iteration met in finding bottom ring path: {unordered_verts} --- {ordered_verts}")
        if ordered_verts[-1] != end:
            print(f"ordered verts final: {ordered_verts}")
            print(f"unordered verts final: {unordered_verts}")
            raise AssertionError(f"Curve verts sorting algorithm did not land in the correct endpoint {end} but in {ordered_verts[-1]}")
        # get coords
        ordered_verts_coords = [
            Vector(bm.verts[vert].co) for vert in ordered_verts
        ]
        return ordered_verts, ordered_verts_coords
    
    @staticmethod
    def get_visible_contour_edges(obj: Object) -> List[int]:
        '''
        Get visible edges. 
        Select outmost contours by select boundary loop, 
        return selected edges' indexes.
        '''
        DRUtils.get_visibles(obj)  # leaves visible edges selected
        bpy.ops.object.editmode_toggle()
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bpy.ops.mesh.select_mode(type="EDGE")
        bpy.ops.mesh.region_to_loop()
        obj.update_from_editmode()
        selected_edges = [e.index for e in bm.edges if e.select]
        bpy.ops.object.editmode_toggle()
        return selected_edges
    
    @staticmethod
    def curve_as_freestyle_edge(obj: Object) -> Object:
        '''
        Solidify the input object (bezier curve) by converting into mesh, extrude edge to face and face to solid. 
        Mark the original edge as freestyle edge. 
        Returns the object. 
        '''
        # select this one only
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.convert(target="MESH")
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.select_mode(type='EDGE')
        bpy.ops.mesh.select_all(action='SELECT')
        obj = bpy.context.object
        bm = bmesh.from_edit_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        original_edges = [e for e in bm.edges if e.select]
        original_edge_index = [e.index for e in original_edges]
        bmesh.ops.extrude_edge_only(bm, edges=bm.edges)  # leaves new edges selected
        bm.edges.ensure_lookup_table()
        bmesh.update_edit_mesh(obj.data)
        new_edges = [e for e in bm.edges if not e.select]
        for e in new_edges:
            v0, v1 = e.verts
            v0.co += Vector((0, -0.001, 0))
            v1.co += Vector((0, -0.001, 0))
        # mark freestyle edge on original edge
        bpy.ops.mesh.select_mode(type='EDGE')
        bpy.ops.mesh.select_all(action='DESELECT')
        bm.edges.ensure_lookup_table()
        for e in original_edge_index:
            bm.edges[e].select_set(True)
        bpy.ops.mesh.mark_freestyle_edge()
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.editmode_toggle()
        return obj
    
    @staticmethod
    def mark_curves_as_freestyle(curves: List[Object]) -> List[Object]:
        '''
        Receive bezier curves, solidify them and mark their edges, return them. 
        '''
        extruded_objects = []
        for curve in curves:
            extruded_objects.append(
                DRUtils.curve_as_freestyle_edge(curve)
            )
        return extruded_objects


class Dedicated_Renderer():
    '''
    Base class for dedicated renderers. 
    '''
    def __init__(self) -> None:
        pass

    def render_object(self, obj: Object, img_path: str, 
                      de_reg: DRUtils.DeRegLevels=DRUtils.DeRegLevels.MEDIUM):
        '''
        Spawn curves representing visible and sharp edges, hide the object and take viewport render, save image and clean up. 
        '''
        raise NotImplementedError(f"Base method should not be called")
    
    def render_model(self, obj: Object, img_path: str):
        bpy.context.scene.render.filepath = img_path
        bpy.ops.render.opengl(write_still=True)
        self.clean_up(obj, [])
    
    def _render_obj_debug(self, obj: Object, img_title: str):
        '''
        render the object's plain, edges versions separately. 
        '''
        raise NotImplementedError(f"Base method should not be called")

    def render_image_and_save_as(self, obj: Object, img_path: str) -> None:
        # TODO: changing to camera render (with freestyle)
        # obj.hide_set(True)  # viewport hiding, since we are capturing viewport, although the viewport is cam's view...
        obj.hide_render = True
        bpy.context.scene.render.filepath = img_path
        # bpy.ops.render.opengl(write_still=True)
        bpy.ops.render.render(write_still=True)
    
    def clean_up(self, obj: Object, curves: List[Object]) -> None:
        bpy.data.objects.remove(obj)
        # if curves == []:
        #     return
        for curve in curves:
            bpy.data.objects.remove(curve)


class DRStraight(Dedicated_Renderer):
    '''
    Dedicated renderer pipeline for objects consisting of only straight lines. 
    '''
    def __init__(self) -> None:
        super().__init__()
    
    def _find_terminal_edges(self, edges: List[BMEdge]) -> Tuple[BMEdge, BMEdge]:
        for edge in edges:
            for vert in edge.verts:
                for link_edge in vert.link_edges:
                    if link_edge not in edges:
                        # next edge: the other vert's other edge
                        return edge, [e for e in [v for v in edge.verts if v != vert][0].link_edges if e != edge][0]
    
    def _sort_subedges(self, edges: List[BMEdge], bm: BMesh) -> List[BMEdge]:
        '''
        Sort given bm edges. Return BMEdges in order. 
        Assumption: a whole edge. edges subdivided at least 1 time. (two edges, 3 verts)
        1. Find end point: aka terminal edge
            endpoint: edge that has one vert with only one link_edge in given edges list
            doesn't matter start or end
        2. starting from one end point (edge): 
            next vert: the vert with both link_edges in list
            add the edge from next_vert.link_edges that's not the same as this one. 
            add vert to "used verts"
        3. Then: 
            edge
            edge.verts, next vert: the vert that's not in used verts
            add edge from vert.link_edges not same
            if only one link_edge in list: terminate. 
        '''
        if len(edges) < 2:
            raise AssertionError(f"Edge not subdivided properly: {edges}")
        # unordered_edges = edges[:]
        terminal_edge, next_edge = self._find_terminal_edges(edges)
        sorted_edges = [terminal_edge, next_edge]
        _used_verts = set(terminal_edge.verts)
        # unordered_edges.remove(terminal_edge)
        for _ in range(len(edges) * 2):  # fake while loop with safety margin
            cur_edge = sorted_edges[-1]
            next_vert = [v for v in cur_edge.verts if v not in _used_verts][0]
            next_edge = [e for e in next_vert.link_edges if e != cur_edge][0]
            if next_edge not in edges:
                break
            sorted_edges.append(next_edge)
            _used_verts.add(next_vert)
        return sorted_edges
    
    def _get_terminal_verts_from(self, sorted_edges: List[BMEdge]) -> Tuple[int, int]:
        '''
        Never empty list
        '''
        v0, v1 = sorted_edges[0].verts
        if len(sorted_edges) < 2:
            return v0.index, v1.index
        if v0 in sorted_edges[1].verts:
            starting_vert = v1
        elif v1 in sorted_edges[1].verts:
            starting_vert = v0
        else:
            raise AssertionError(f"starting edges not sorted: {sorted_edges}")
        v2, v3 = sorted_edges[-1].verts
        if v2 in sorted_edges[-2].verts:
            ending_vert = v3
        elif v3 in sorted_edges[-2].verts:
            ending_vert = v2
        else:
            raise AssertionError(f"ending edges not sorted: {sorted_edges}")
        return starting_vert.index, ending_vert.index
    
    def edge_filtering(self, obj: Object) -> List[Tuple[int, int]]:
        '''
        Get visible edges. 
        Assumption: All straight lines. 
        Note: subdivide each not-fully-visible edge to 10 segments to get an
        more accurate line render. 
        Returns list of vert pairs, the start and end of each edge to be rendered. Index. 
        '''
        edge_as_vert_pairs = []
        _, sharp_edges, _ = DRUtils.get_sharps(obj)
        _, visible_edges, _ = DRUtils.get_visibles(obj, mode='EDGE')
        visible_verts, _, _ = DRUtils.get_visibles(obj, mode='VERT') # get verts and edges separately
        # get visible + sharp edges: visharp
        visharp_edges = list(set(sharp_edges).intersection(visible_edges))
        # visibility check: full vs. partial
        bpy.ops.object.editmode_toggle()
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bpy.ops.mesh.select_mode(type='EDGE')
        bpy.ops.mesh.select_all(action='DESELECT')
        bm.edges.ensure_lookup_table()
        # _i = 0
        for edge_index in visharp_edges:
            # _i += 1
            edge: BMEdge = bm.edges[edge_index]
            vert0, vert1 = edge.verts
            if vert0.index in visible_verts and vert1.index in visible_verts:
                edge_as_vert_pairs.append(
                    (vert0.index, vert1.index)  # and order does not matter
                )
                continue  # fully visible, no need to subdivide, otherwise: 
            # subdivide
            # TODO: make cuts relative to edge length
            bpy.ops.mesh.select_mode(type='EDGE')
            bpy.ops.mesh.select_all(action='DESELECT')  # deselect again, since previous loop will leave mode in verts and verts selected. 
            edge.select_set(True)  # TODO: either select it here or examine the output of subdivide_edges
            obj.update_from_editmode()
            bmesh.ops.subdivide_edges(bm, edges=[edge], cuts=10) # leaves the edges selected
            bm.edges.ensure_lookup_table()
            bmesh.update_edit_mesh(me)
            # get edges and sort
            sub_edges = [e for e in bm.edges if e.select]
            sorted_subedges = self._sort_subedges(sub_edges, bm)
            bm.edges.ensure_lookup_table()
            # one more edges visibility check
            _, new_vis_edges, _ = DRUtils.get_visibles(obj, mode='VERT', bm=bm)
            # intersection with subedges to get visible subedges
            visible_subedges = [edge for edge in sorted_subedges if edge.index in new_vis_edges]
            # print(f"sorted_subedges: {sorted_subedges}")
            # print(f"new_vis_edges: {new_vis_edges}")
            # print(f"visible subedges: {visible_subedges}")
            if visible_subedges == []:  # completely hidden, not rendered
                continue
            # find continuous segments
            segment = []
            while visible_subedges != []:
                cur_subedge = visible_subedges.pop(0)
                # test continuity
                # -- init
                if segment == []:
                    segment.append(cur_subedge)
                    continue
                if any([cur_subedge in vert.link_edges for vert in segment[-1].verts]):
                    # print(f"added next subedge: {cur_subedge}")
                    segment.append(cur_subedge)
                else:
                    # print(f"current segment terminates at: {cur_subedge}")
                    edge_as_vert_pairs.append(
                        self._get_terminal_verts_from(segment)
                    )
                    # print(f"added segment: {segment}")
                    segment = []
            if segment != []:  # the last segment, ending peacefully
                # print(f"added last segment: {segment}")
                edge_as_vert_pairs.append(
                    self._get_terminal_verts_from(segment)
                )
            # if _i >= 2:  # some debugger... 
            #     raise AssertionError()
        bpy.ops.object.editmode_toggle()
        return edge_as_vert_pairs
    
    def render_object(self, obj: Object, img_path: str, 
                      de_reg: DRUtils.DeRegLevels=DRUtils.DeRegLevels.MEDIUM, obj_name: str=None) -> None:
        obj_name = obj.name if not obj_name else obj_name
        edges = self.edge_filtering(obj)
        disturbed_coords = DRUtils.disturb_edges(edges, obj, de_reg)
        spawned_curves = DRUtils.spawn_curves(disturbed_coords, obj_name)
        extruded_curves = DRUtils.mark_curves_as_freestyle(spawned_curves)
        self.render_image_and_save_as(obj, img_path)
        # self.clean_up(obj, extruded_curves)
    
    def obj_to_curves_only(self, obj: Object, de_reg: DRUtils.DeRegLevels=DRUtils.DeRegLevels.MEDIUM, obj_name: str=None) -> List[Object]:
        obj_name = obj.name if not obj_name else obj_name
        edges = self.edge_filtering(obj)
        disturbed_coords = DRUtils.disturb_edges(edges, obj, de_reg)
        spawned_curves = DRUtils.spawn_curves(disturbed_coords, obj_name)
        return spawned_curves
    
    def _render_obj_debug(self, obj: Object, img_title: str):
        # take plain shot
        bpy.context.scene.render.filepath = img_title + "_obj.png"
        bpy.ops.render.opengl(write_still=True)
        # edges version
        self.render_object(obj, img_title + "_edges.png")
    

class DRCylindrical(Dedicated_Renderer):
    '''
    Dedicated renderer pipeline for objects consisting of cylinders. 
    '''
    def __init__(self) -> None:
        super().__init__()

    def edge_filtering(self, obj: Object) -> Tuple[List[Tuple[int, int]], List[List[Vector]]]:
        '''
        For a cylinder, get its upper circle, bottom ring and two vertical edges on contour. 
        Returns straight edges (list of edge in vert index pair), 
        and curves (list of curve in vertex coordinates in Vector). 
        Ordered. 
        '''
        circle_and_ring = []
        vertical_contour_edges = []
        # prep
        sharp_verts, sharp_edges, sharp_faces = DRUtils.get_sharps(obj)
        visible_verts, visible_edges, visible_faces = DRUtils.get_visibles(obj)
        # print(f"sharp edges: {sharp_edges}")
        # print(f"visible edges: {visible_edges}")
        contour_edges = DRUtils.get_visible_contour_edges(obj)
        # sharp - invisible = upper circle and bottom ring
        visharp_edges = [e for e in sharp_edges if e in visible_edges]
        # verti-contour = contour edge not visharp
        vertical_contour_edge_indexes = [e for e in contour_edges if e not in visharp_edges]
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.select_mode(type='EDGE')
        bpy.ops.mesh.select_all(action='DESELECT')
        bm = bmesh.from_edit_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        # get edge endpoints
        for edge_index in vertical_contour_edge_indexes:
            edge = bm.edges[edge_index]
            vertical_contour_edges.append(
                (edge.verts[0].index, edge.verts[1].index)
            )
        # separate circle and ring from edge loop
        _edge = visharp_edges[0]  # random vs. fixed: no difference... random makes debug a lil harder
        bm.edges[_edge].select_set(True)
        bpy.ops.mesh.loop_multi_select(ring=False)
        first_curve_edges: List[BMEdge] = [e for e in bm.edges if e.select]
        # leave only parts that's also visharp
        first_curve_edge_indexes = [e.index for e in first_curve_edges if e.index in visharp_edges]
        other_curve_edge_indexes = [e for e in visharp_edges if e not in first_curve_edge_indexes]
        first_curve_verts, first_curve_coords = DRUtils.get_curve_points(first_curve_edge_indexes, bm)
        other_curve_verts, other_curve_coords = DRUtils.get_curve_points(other_curve_edge_indexes, bm)
        circle_and_ring = [first_curve_coords, other_curve_coords]
        bpy.ops.object.editmode_toggle()
        return vertical_contour_edges, circle_and_ring
    
    def render_object(self, obj: Object, img_path: str, de_reg: DRUtils.DeRegLevels = DRUtils.DeRegLevels.MEDIUM, obj_name: str=None):
        obj_name = obj.name if not obj_name else obj_name
        vertical_contour_edges, curves = self.edge_filtering(obj)
        disturbed_edges = DRUtils.disturb_edges(vertical_contour_edges, obj, de_reg)
        disturbed_curves = DRUtils.disturb_curves(curves, obj, de_reg)
        spawned_curves = DRUtils.spawn_curves(disturbed_edges + disturbed_curves, obj_name)
        extruded_curves = DRUtils.mark_curves_as_freestyle(spawned_curves)
        self.render_image_and_save_as(obj, img_path)
        self.clean_up(obj, extruded_curves)

    def obj_to_curves_only(self, obj: Object, de_reg: DRUtils.DeRegLevels=DRUtils.DeRegLevels.MEDIUM, obj_name: str=None) -> List[Object]:
        obj_name = obj.name if not obj_name else obj_name
        vertical_contour_edges, curves = self.edge_filtering(obj)
        disturbed_edges = DRUtils.disturb_edges(vertical_contour_edges, obj, de_reg)
        disturbed_curves = DRUtils.disturb_curves(curves, obj)
        spawned_curves = DRUtils.spawn_curves(disturbed_edges + disturbed_curves, obj_name)
        return spawned_curves
    
    def _render_obj_debug(self, obj: Object, img_title: str):
        # take plain shot
        bpy.context.scene.render.filepath = img_title + "_obj.png"
        bpy.ops.render.opengl(write_still=True)
        # edges version
        self.render_object(obj, img_title + "_edges.png")
    

class DRSphered(Dedicated_Renderer):
    '''
    Dedicated renderer pipeline for objects consisting of spheres. 
    '''
    def __init__(self) -> None:
        super().__init__()

    def _get_up_curve(self, visharps: List[int], bm: BMesh) -> Tuple[List[int], List[Vector]]:
        '''
        Obtain the curve going from one endpoint of visible half bottom ring, 
        through the top vertex, to the other endpoint of the ring. 
        Returns an ordered list of int, index of verts of the curve, and corresponding Vector coords. ordered. 
        Note: does not touch edit mode toggles - should always be under the same context/bm as get_bottom_curve
        It can be modified to operates its own bm tho, but in case it returns different indexes...
        '''
        bpy.ops.mesh.select_mode(type='VERT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bm.verts.ensure_lookup_table()
        # get endpoints
        start, end = DRUtils.find_curve_endpoints(visharps, DRUtils._edges_from_verts(visharps, bm), bm)  # yeah weeks later it still sometimes unhappy
        # print(start)
        # print(len(bm.verts[start].link_edges))
        # # test by selection: 
        # verts = [start]
        # edges = [e.index for e in bm.verts[start].link_edges]
        # bm.edges.ensure_lookup_table()
        # bpy.ops.mesh.select_mode(type='EDGE')
        # bpy.ops.mesh.select_all(action="DESELECT")
        # for edge in bm.edges:
        #     if edge.index in edges:
        #         edge.select_set(True)
        # bpy.context.object.update_from_editmode()
        # raise
        # get linked edges of start and end vertex
        start_edges = bm.verts[start].link_edges
        end_edges = bm.verts[end].link_edges
        # print(f"start vert's edges: {len(start_edges)}")  # 2 here
        # print(f"end vert's edges: {len(end_edges)}")
        if len(start_edges) > 3 or len(end_edges) > 3:
            raise AssertionError(f"Failed to find start/end point of up curve on bottom ring, consider increasing threshold value: {start_edges} --- {end_edges}")
        # use face angle of the edge to distinguish the one pointing up
        # (the other edges are on the bottom face, so the edge angle should be similar)
        se_angles = []
        for se in start_edges:
            se_angles.append(
                (se, se.calc_face_angle())
            )
        ee_angles = []
        for ee in end_edges:
            ee_angles.append(
                (ee, ee.calc_face_angle())
            )
        se_up = DRUtils._find_diff_edge(se_angles)
        ee_up = DRUtils._find_diff_edge(ee_angles)
        # select the two edges, and then the edge loop
        # to form a ordered vertices list
        bpy.ops.mesh.select_mode(type='EDGE')
        bpy.ops.mesh.select_all(action='DESELECT')
        # first half
        bm.edges.ensure_lookup_table()
        bm.edges[se_up[0].index].select = True
        bpy.ops.mesh.loop_multi_select(ring=False)
        up_verts = [v for v in bm.verts if v.select]  # now list of BMVert
        up_verts.sort(key=DRUtils._sort_by_z)  # sort by z value, low to high
        # other half
        bpy.ops.mesh.select_all(action='DESELECT')
        bm.edges[ee_up[0].index].select = True
        bpy.ops.mesh.loop_multi_select(ring=False)
        down_verts = [v for v in bm.verts if v.select]  # now list of BMVert
        down_verts.sort(key=DRUtils._sort_by_z, reverse=True)  # high to low
        # combine the curve, remove the repeated top vert
        curve = up_verts + down_verts[1:]
        # leave only index... ugly, but for a unified spawn pipeline... 
        curve_ids = [v.index for v in curve]
        curve_coords = [v.co for v in curve]
        return curve_ids, curve_coords
    
    def edge_filtering(self, obj: Object) -> List[List[Vector]]:
        '''
        the specialized edge filtering pipeline for half sphere: 
        visible half bottom ring + curve from endpoint to top to the other endpoint. 
        returns list of curves. curve: list of vert coordinates
        '''
        # get visible half of bottom ring through visharp filtering
        sharp_verts, sharp_edges, _ = DRUtils.get_sharps(obj, threshold=1.570796)  # 90 degrees
        visible_verts, visible_edges, _ = DRUtils.get_visibles(obj)
        # TODO: test if list-set-list works the same as list comprehension
        visharp_verts = list(set(sharp_verts).intersection(visible_verts))
        visharp_edges = list(set(sharp_edges).intersection(visible_edges))
        # get bmesh
        bpy.ops.object.editmode_toggle()
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        bm.verts.ensure_lookup_table()  # TODO: optimize calls to these methods
        bm.edges.ensure_lookup_table()
        # two curves; these two methods should operate under same context/bm
        upper_curve_verts, upper_curve_coords = self._get_up_curve(visharp_verts, bm)
        bottom_curve_verts, bottom_curve_coords = DRUtils.get_curve_points(visharp_edges, bm)
        # # get edges from verts
        # upper_curve_edges = DRUtils._edges_from_verts(upper_curve_verts, bm)
        # bottom_curve_edges = DRUtils._edges_from_verts(bottom_curve_verts, bm)
        bpy.ops.object.editmode_toggle()
        return [upper_curve_coords, bottom_curve_coords]

    def render_object(self, obj: Object, img_path: str, de_reg: DRUtils.DeRegLevels = DRUtils.DeRegLevels.MEDIUM, obj_name: str=None):
        obj_name = obj.name if not obj_name else obj_name
        curves = self.edge_filtering(obj)
        disturbed_coords = DRUtils.disturb_curves(curves, obj, de_reg)
        spawned_curves = DRUtils.spawn_curves(disturbed_coords, obj_name)
        extruded_curves = DRUtils.mark_curves_as_freestyle(spawned_curves)
        self.render_image_and_save_as(obj, img_path)
        self.clean_up(obj, extruded_curves)

    def obj_to_curves_only(self, obj: Object, de_reg: DRUtils.DeRegLevels=DRUtils.DeRegLevels.MEDIUM, obj_name: str=None) -> List[Object]:
        obj_name = obj.name if not obj_name else obj_name
        curves = self.edge_filtering(obj)
        disturbed_coords = DRUtils.disturb_curves(curves, obj, de_reg)
        spawned_curves = DRUtils.spawn_curves(disturbed_coords, obj_name)
        return spawned_curves
    
    def _render_obj_debug(self, obj: Object, img_title: str):
        # take plain shot
        bpy.context.scene.render.filepath = img_title + "_obj.png"
        bpy.ops.render.opengl(write_still=True)
        # edges version
        self.render_object(obj, img_title + "_edges.png")


# class DRController():
#     def __init__(self, dataset_name: str, paramcsv_name: str=None, override_path: str=None) -> None:
#         self._dataset_name = dataset_name
#         self.paramcsv_name = paramcsv_name if paramcsv_name else None
#         self.img_folder, self.params = DRUtils.read_dataset(dataset_name, self.paramcsv_name, override_path)
#         DRUtils.set_up_blender()
#         self.straight_renderer = DRStraight()
#         self.cylindrical_renderer = DRCylindrical()
#         self.sphered_renderer = DRSphered()

#     def get_dedicated_renderer(self, comp: Component) -> Dedicated_Renderer:
#         '''
#         Return the right renderer according to type declarations. 
#         '''
#         type_dec = (comp.type, comp.subtype)
#         if type_dec in DRUtils.straight_comps:
#             return self.straight_renderer
#         elif type_dec in DRUtils.cylindrical_comps:
#             return self.cylindrical_renderer
#         elif type_dec in DRUtils.sphered_comps:
#             return self.sphered_renderer
#         else:
#             raise ValueError(f"Unexpected type combination: {type_dec} from {comp}")

#     def render_dataset(self, plain_model_shot: bool=False):
#         '''
#         Render the images for the whole dataset according to its csv description file. 

#         @params
#         plain_model_shot: enable to capture purely the 3D model, without any sketch effects
#         '''
#         if plain_model_shot:
#             for row in self.params:
#                 # let it throw potential index out of bounds here, instead of in SRP_Spawn_Convert
#                 rname, rtype, rsubtype = row[:3]
#                 rparams = row[3:]
#                 component, obj = DRUtils.spawn(
#                     rtype, rsubtype, rparams, obj_name=rname
#                 )
#                 renderer = self.get_dedicated_renderer(component)
#                 filepath = os.path.join(self.img_folder, rname)
#                 renderer.render_model(obj, filepath)
#         else:
#             for row in self.params:
#                 # let it throw potential index out of bounds here, instead of in SRP_Spawn_Convert
#                 rname, rtype, rsubtype = row[:3]
#                 rparams = row[3:]
#                 component, obj = DRUtils.spawn(
#                     rtype, rsubtype, rparams, obj_name=rname
#                 )
#                 renderer = self.get_dedicated_renderer(component)
#                 filepath = os.path.join(self.img_folder, rname)
#                 de_reg = random.choice(list(DRUtils.DeRegLevels)[1:])  # perfect and light too similar
#                 renderer.render_object(obj, filepath, de_reg)
    
#     def render_single_shot(self, img_name: str):
#         '''
#         Render one single image in the dataset, mostly for debugging purposes. 
#         '''
#         entry = None
#         for row in self.params:
#             if row[0] == img_name:
#                 entry = row
#                 break
#         if entry is None:
#             raise AssertionError(f"Target img not found in dataset: {img_name} in {self._dataset_name}")
#         rname, rtype, rsubtype = row[:3]
#         rparams = row[3:]
#         # print(rparams[:3])
#         # print(rparams[3:6])
#         # print(rparams[6:9])
#         # print(rparams[9:12])
#         # print(rparams[12:15])
#         # print(rparams[15:18])
#         # print(rparams[18:21])
#         # print(rparams[21:24])
#         # print(rparams[24:27])
#         component, obj = DRUtils.spawn(
#             rtype, rsubtype, rparams, obj_name=rname
#         )
#         renderer = self.get_dedicated_renderer(component)
#         filepath = os.path.join(IOHandler.single_shot_folder, rname)
#         renderer._render_obj_debug(obj, filepath[:-4])


if __name__ == "__main__":
    renderer = DRStraight()
    obj = bpy.context.active_object
    path = "./renders/straight_test.png"
    renderer.render_object(obj, path)
