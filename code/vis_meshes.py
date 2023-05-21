"""
Visualize 3D meshes in aitviewer.
"""
import numpy as np
import os.path as osp
import glob
import argparse
import trimesh

from aitviewer.viewer import Viewer
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.renderables.meshes import VariableTopologyMeshes


def main(data_root):
    cols, rows = 800, 1200
    # if want to save video, use the following line and line 34
    v = HeadlessRenderer(size=(cols, rows))
    # if want to visualize, use the following line and line 35
    # v = Viewer(size=(cols, rows))
    v.playback_fps = 30.0
    v.scene.camera.fov = 45.0
    v.scene.camera.position[0] = 0.072
    v.scene.camera.position[1] = -0.365
    v.scene.camera.position[2] = 2.915
    v.scene.camera.target[0] = 0.015
    v.scene.camera.target[1] = -0.423
    v.scene.camera.target[2] = 0.044

    # Example 1: Load 3D scans (.pkl format)
    pkl_folder = osp.join(data_root, 'meshes_pkl')
    if osp.exists(pkl_folder):
        meshes = VariableTopologyMeshes.from_directory(pkl_folder, name='Scan Mesh')
        v.scene.add(meshes)
    
    # Example 2: Load 3D scans (.ply format)
    if osp.exists(data_root):
        smplx_meshes_names = sorted(glob.glob(osp.join(data_root, '*.ply')))
        smplx_meshes = VariableTopologyMeshes.from_plys(smplx_meshes_names, name='Scan Mesh')
        v.scene.add(smplx_meshes)

    v.save_video(video_dir=osp.join(data_root, "vis.mp4"))
    # v.run()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/path/to/your/data/folder')
    args = parser.parse_args()
    
    main(data_root=args.data_root)
