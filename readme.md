Install python environment: 
 - `conda create -n "distortion" python=3.10`
 - `pip install -r ./requirements.txt`

Activate conda environment through `conda activate distortion`

Download blender 3.2 from https://download.blender.org/release/Blender3.2/ and unzip into `./blender`; 

Remove blender's bundled python folder at `./blender/blender-3.2.2-windows.x64/3.2/python` and start blender with conda env activated
 - or copy paste conda's python env folder (e.g. envs/distortion) into ./3.2 and rename it to python; then remove the freestyle folder at `3.2/python/Lib/site-packages/freestyle` to make blender fallback to its own freestyle
