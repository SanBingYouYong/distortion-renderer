import bpy
from bpy.types import Context

# import typing
import os
import sys
from pathlib import Path
import shutil
# from subprocess import Popen, PIPE
# import logging
import numpy as np

# import local modules
file = Path(__file__).resolve()
parent = file.parents[0]  # 1 if run manually, 0 if run from driver.py
print(f"parent: {parent}")
sys.path.append(str(parent))

# from params import DAGParams
# from paramgen import DAGParamGenerator
# from paramload import DAGParamLoader
from distortion import Dedicated_Renderer, DRStraight, DRCylindrical, DRSphered, DRUtils


def find_curves(obj_name: str, name_only: bool = True):
    '''
    Find curves related to the object with the given name

    Parameters:
    obj_name: str
        The name of the object
    name_instead_of_obj: bool
        If True, return the names of the curves; otherwise, return the curves themselves as Objects and their names
    '''
    curves = []
    curve_names = []
    for obj in bpy.data.objects:
        if obj.type == 'CURVE' and obj_name in obj.name and obj.name != obj_name and "BezierCurve_" in obj.name:
            curves.append(obj)
            curve_names.append(obj.name)
    if name_only:
        return curve_names
    return curves, curve_names

def find_all_curves(name_only: bool = True):
    '''
    Find all curves. 

    Parameters:
    name_instead_of_obj: bool
        If True, return the names of the curves; otherwise, return the curves themselves as Objects and their names
    '''
    curves = []
    curve_names = []
    for obj in bpy.data.objects:
        if obj.type == 'CURVE' and "BezierCurve_" in obj.name:
            curves.append(obj)
            curve_names.append(obj.name)
    if name_only:
        return curve_names
    return curves, curve_names

def delete_objects(obj_names: list):
    '''
    Delete all objects with the given names
    '''
    objects = bpy.data.objects
    for obj_name in obj_names:
        if obj_name in objects:
            objects.remove(objects[obj_name])

def get_renderer(dr_type: str) -> Dedicated_Renderer:
    '''
    Get the renderer based on the given type

    Parameters:
    dr_type: str
        The type of the renderer

    Returns:
    renderer: Dedicated_Renderer
        The renderer object
    '''
    renderer = None
    if dr_type == 'dr_straight':
        renderer = DRStraight()
    elif dr_type == 'dr_cylinder':
        renderer = DRCylindrical()
    elif dr_type == 'dr_sphere':
        renderer = DRSphered()
    else:
        raise ValueError(f"Invalid DR type: {dr_type}")
    return renderer

def get_de_reg(de_reg: str) -> DRUtils.DeRegLevels:
    '''
    Get the DeReg level based on the given string

    Parameters:
    de_reg: str
        The string representation of the DeReg level

    Returns:
    de_reg: DRUtils.DeRegLevels
        The DeReg level
    '''
    if de_reg == 'dl_perfect':
        return DRUtils.DeRegLevels.PERFECT
    elif de_reg == 'dl_light':
        return DRUtils.DeRegLevels.LIGHT
    elif de_reg == 'dl_medium':
        return DRUtils.DeRegLevels.MEDIUM
    elif de_reg == 'dl_heavy':
        return DRUtils.DeRegLevels.HEAVY
    else:
        raise ValueError(f"Invalid distortion level: {de_reg}")


class DistortionOperator(bpy.types.Operator):
    bl_idname = "object.distortion_operator"
    bl_label = "Distort Image"

    def execute(self, context):
        print(f"You've called {self.bl_label}")
        dr_type = context.scene.dr_type
        dl = context.scene.distort_level
        obj = bpy.context.active_object  # no need to hide, since editmode will take care of visibility on single obj
        if obj is None:
            self.report({'ERROR'}, "No object selected")
            return {'CANCELLED'}
        bpy.ops.object.duplicate()  # create duplicate to avoid affecting original object
        dup = bpy.context.active_object
        renderer = get_renderer(dr_type)
        de_reg = get_de_reg(dl)
        spawned_curves = renderer.obj_to_curves_only(dup, de_reg=de_reg, obj_name=obj.name)
        context.scene.last_distorted_object_name = obj.name
        delete_objects([dup.name])
        if context.scene.remove_obj_after_spawn:
            delete_objects([obj.name])
        return {'FINISHED'}

class RenderAsImage(bpy.types.Operator):
    bl_idname = "object.render_as_image"
    bl_label = "Render as Image"

    def execute(self, context):
        print(f"You've called {self.bl_label}")

        # FIND CURVES
        curves = []
        if context.scene.render_all_curves:
            curves, curve_names = find_all_curves(name_only=False)
        else:
            curves, curve_names = find_curves(context.scene.last_distorted_object_name, name_only=False)

        # VISIBILITY
        temp_hidden_objs = []
        if context.scene.render_curves_only:
            # hide all other objects
            for obj in bpy.data.objects:
                if obj.name not in curve_names:
                    # obj.hide_set(True)
                    obj.hide_render = True
                    temp_hidden_objs.append(obj)

        # PATH
        img_path = context.scene.image_path
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = img_path

        # FREESTYLE
        freestyle = context.scene.use_freestyle
        if freestyle:
            bpy.context.scene.render.use_freestyle = True
            curves = DRUtils.mark_curves_as_freestyle(curves)
        else:
            bpy.context.scene.render.use_freestyle = False
        
        # RENDER
        viewport = context.scene.viewport_render
        if viewport and freestyle:
            self.report({'WARNING'}, "Freestyle is not supported in viewport render")
        if viewport:
            original_engine = bpy.context.scene.render.engine
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'
            bpy.ops.render.opengl(write_still=True)
            bpy.context.scene.render.engine = original_engine
        else:
            original_engine = bpy.context.scene.render.engine
            bpy.context.scene.render.engine = 'CYCLES'
            bpy.ops.render.render(write_still=True)
            bpy.context.scene.render.engine = original_engine
        
        # CLEAN UP
        if context.scene.render_curves_only:
            for obj in temp_hidden_objs:
                # obj.hide_set(False)
                obj.hide_render = False
        delete_objects(curve_names)

        return {'FINISHED'}
    
class DistortAndRender(bpy.types.Operator):
    bl_idname = "object.distort_and_render"
    bl_label = "Distort and Render"

    def execute(self, context):
        print(f"You've called {self.bl_label}")
        bpy.ops.object.distortion_operator()
        bpy.ops.object.render_as_image()  # should we lock this to default settings?
        return {'FINISHED'}
    
class DeleteCurrentCurves(bpy.types.Operator):
    bl_idname = "object.delete_current_curves"
    bl_label = "Delete Current Curves"

    def execute(self, context):
        print(f"You've called {self.bl_label}")
        if context.scene.last_distorted_object_name == "":
            self.report({'ERROR'}, "No object has been distorted yet")
            return {'CANCELLED'}
        obj = context.active_object
        if obj is None:
            obj_name = context.scene.last_distorted_object_name
        else:
            obj_name = obj.name
        curves = find_curves(obj_name)
        delete_objects(curves)
        return {'FINISHED'}

class DeleteAllCurves(bpy.types.Operator):
    bl_idname = "object.delete_all_curves"
    bl_label = "Delete All Curves"

    def execute(self, context):
        print(f"You've called {self.bl_label}")
        curves = find_all_curves()
        delete_objects(curves)
        return {'FINISHED'}

class ToggleCameraView(bpy.types.Operator):
    bl_idname = "object.toggle_camera_view"
    bl_label = "Toggle Camera View"

    def execute(self, context):
        print(f"You've called {self.bl_label}")
        bpy.ops.view3d.view_camera()
        return {'FINISHED'}

class ShowAllObjects(bpy.types.Operator):
    bl_idname = "object.show_all_objects"
    bl_label = "Show All Objects"

    def execute(self, context):
        print(f"You've called {self.bl_label}")
        for obj in bpy.data.objects:
            obj.hide_set(False)
        return {'FINISHED'}


class DistortionPanel(bpy.types.Panel):
    bl_label = "Distortion Panel"
    bl_idname = "SIUI_PT_DistortionPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # View
        box = layout.box()
        box.label(text="View", icon='CAMERA_STEREO')

        box.prop(scene, "show_current_shape", text="Show Current Shape")
        box.prop(scene, "show_spawned_curves", text="Show Spawned Curves")
        box.operator("object.show_all_objects", text="Show All Objects")
        
        box.prop(scene, "cam_angle_hori", text="Camera Horizontal Angle")
        box.prop(scene, "cam_angle_vert", text="Camera Vertical Angle")
        box.operator("object.toggle_camera_view", text="Toggle Camera View")

        # Manipulate
        box = layout.box()
        box.label(text="Manipulate", icon='MODIFIER_DATA')
        box.operator("object.delete_current_curves", text="Delete Current Curves")
        box.operator("object.delete_all_curves", text="Delete All Curves")
        
        # Combined
        box = layout.box()
        box.label(text="Combined", icon='RNA_ADD')
        box.operator("object.distort_and_render", text="Distort and Render")

        # Distort Into Curves
        box = layout.box()
        box.label(text="Distort Into Curves", icon='OUTLINER_DATA_CURVES')
        box.prop(scene, "remove_obj_after_spawn", text="Remove Object After Spawn")
        box.prop(scene, "dr_type", text="Object Type")
        box.prop(scene, "distort_level", text="Distortion Level")
        box.operator("object.distortion_operator", text="Distort Into Curves")

        # Render as Image
        box = layout.box()
        box.label(text="Render as Image", icon='CAMERA_DATA')
        box.prop(scene, "use_freestyle", text="Use Freestyle")
        box.prop(scene, "viewport_render", text="Viewport Render")
        box.prop(scene, "image_path", text="Image Path")
        box.prop(scene, "render_all_curves", text="Render All Curves")
        box.prop(scene, "render_curves_only", text="Render Curves Only")
        box.operator("object.render_as_image", text="Render as Image")


def update_current_shape_visibility(self, context):
    obj = bpy.data.objects.get(context.scene.last_distorted_object_name) if context.scene.last_distorted_object_name != "" else None
    if obj is None:
        print("No object has been distorted yet or it has been removed")
        return
    if context.scene.show_current_shape:
        obj.hide_set(False)
    else:
        obj.hide_set(True)

def update_spawned_curves_visibility(self, context):
    if context.scene.show_spawned_curves:
        curves, _ = find_all_curves(name_only=False)
        for curve in curves:
            curve.hide_set(False)
    else:
        curves, _ = find_all_curves(name_only=False)
        for curve in curves:
            curve.hide_set(True)

def update_camera_viewing_angle(self, context):
    scene = context.scene
    new_lr_angle = scene.cam_angle_hori
    new_down_angle = scene.cam_angle_vert
    # sin(h/2) = angle
    height = 2 * np.sin(np.deg2rad(new_down_angle))
    # cos(h/2) = radius
    radius = 2 * np.cos(np.deg2rad(new_down_angle))
    camera_track = bpy.data.objects["CameraTrack"]
    camera_track.rotation_euler[2] = np.deg2rad(new_lr_angle)
    camera_track.location[2] = height
    track_radius_scaler = radius / 2
    camera_track.scale = (track_radius_scaler, track_radius_scaler, track_radius_scaler)



def register():
    bpy.types.Scene.dr_type = bpy.props.EnumProperty(
        items=[('dr_straight', 'Straight Line', "Dedicated Distortion Renderer for objects containing only straight edges", 'META_CUBE', 0),
               ('dr_cylinder', 'Cylindrical', "Dedicated Distortion Renderer for cylinders", 'MESH_CYLINDER', 1),
               ('dr_sphere', 'Half-spherical', "Dedicated Distortion Renderer for half-spheres", 'MATSPHERE', 2)],
        name="Object Type",
        description="Select the type of DR to use"
    )
    bpy.types.Scene.distort_level = bpy.props.EnumProperty(
        items=[ ('dl_perfect', 'Perfect', "No distortion", 'SEQUENCE_COLOR_04', 0),
                ('dl_light', 'Light', "Low level of distortion", 'SEQUENCE_COLOR_03', 0),
                ('dl_medium', 'Medium', "Medium level of distortion", 'SEQUENCE_COLOR_02', 1),
                ('dl_heavy', 'Heavy', "High level of distortion", 'SEQUENCE_COLOR_01', 2)],
        name="Distortion Level",
        description="Select the level of distortion"
    )
    # TODO: show/hide both viewport and render? 
    bpy.types.Scene.show_current_shape = bpy.props.BoolProperty(
        name="Show Current Shape",
        description="Show the current shape",
        default=False,
        update=update_current_shape_visibility
    )
    bpy.types.Scene.show_spawned_curves = bpy.props.BoolProperty(
        name="Show Spawned Curves",
        description="Show the spawned curves",
        default=False,
        update=update_spawned_curves_visibility
    )
    bpy.types.Scene.cam_angle_hori = bpy.props.FloatProperty(
        name="Camera Horizontal Angle",
        description="Set the camera horizontal angle",
        default=45.0,
        min=0.0,
        max=90.0,
        update=update_camera_viewing_angle
    )
    bpy.types.Scene.cam_angle_vert = bpy.props.FloatProperty(
        name="Camera Vertical Angle",
        description="Set the camera vertical angle",
        default=30.0, 
        min=0.0,
        max=90.0,
        update=update_camera_viewing_angle
    )
    bpy.types.Scene.remove_obj_after_spawn = bpy.props.BoolProperty(
        name="Remove Object After Spawn",
        description="Remove the object after spawning distorted curves",
        default=False
    )
    bpy.types.Scene.last_distorted_object_name = bpy.props.StringProperty(
        name="Last Distorted Object Name",
        description="Name of the last distorted object",
        default=""
    )
    bpy.types.Scene.image_path = bpy.props.StringProperty(
        name="Image Path",
        subtype='FILE_PATH',
        description="Set the image path",
        default=""
    )
    bpy.types.Scene.render_all_curves = bpy.props.BoolProperty(
        name="Render All Curves",
        description="Render all curves or only last distorted curves",
        default=False
    )
    bpy.types.Scene.render_curves_only = bpy.props.BoolProperty(
        name="Render Curves Only",
        description="Render curves only",
        default=True
    )
    bpy.types.Scene.use_freestyle = bpy.props.BoolProperty(
        name="Use Freestyle",
        description="Use Freestyle",
        default=True
    )
    bpy.types.Scene.viewport_render = bpy.props.BoolProperty(
        name="Viewport Render",
        description="Capture the viewport as an image",
        default=False
    )


    bpy.utils.register_class(DistortionPanel)
    bpy.utils.register_class(DistortionOperator)
    bpy.utils.register_class(RenderAsImage)
    bpy.utils.register_class(DistortAndRender)
    bpy.utils.register_class(DeleteCurrentCurves)
    bpy.utils.register_class(DeleteAllCurves)
    bpy.utils.register_class(ToggleCameraView)
    bpy.utils.register_class(ShowAllObjects)

def unregister():
    bpy.utils.unregister_class(DistortionPanel)
    bpy.utils.unregister_class(DistortionOperator)
    bpy.utils.unregister_class(RenderAsImage)
    bpy.utils.unregister_class(DistortAndRender)
    bpy.utils.unregister_class(DeleteCurrentCurves)
    bpy.utils.unregister_class(DeleteAllCurves)
    bpy.utils.unregister_class(ToggleCameraView)
    bpy.utils.unregister_class(ShowAllObjects)


if __name__ == "__main__":
    register()
