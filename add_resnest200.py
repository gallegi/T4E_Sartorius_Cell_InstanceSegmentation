import shutil
import detectron2
import os

DEST_FOLDER = os.path.dirname(detectron2.__file__)

shutil.copy('resnest200/resneSt.py', f'{DEST_FOLDER}/modeling/backbone/')
shutil.copy('resnest200/splat.py', f'{DEST_FOLDER}/modeling/backbone/')
shutil.copy('resnest200/fpn_resneSt.py', f'{DEST_FOLDER}/modeling/backbone/')