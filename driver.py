import subprocess


BLENDER_PATH = "./blender/blender-3.2.2-windows-x64/blender"
UI_PATH = "./src/ui_distortion.py"
BLEND_FILE_PATH = "./src/plugin_distortion.blend"

def run_blender():
    subprocess.run([BLENDER_PATH, BLEND_FILE_PATH, "--python", UI_PATH])

if __name__ == "__main__":
    run_blender()
