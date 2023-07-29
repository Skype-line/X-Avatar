"""
render the 3D scan with virtual cameras to get RGB-D images
"""
import numpy as np
import os
import tqdm
import argparse
from aitviewer.headless import HeadlessRenderer
from aitviewer.scene.camera import PinholeCamera
from aitviewer.utils.path import circle
from aitviewer.renderables.meshes import VariableTopologyMeshes

def render_color_depth_mask_image(data_root, process_seq=None):
    print('Rendering color and depth images...')
    print('Loading {}'.format(data_root))
    sub_folder_list = sorted(os.listdir(data_root))
    cols, rows = 800, 1200
    viewer = HeadlessRenderer(size=(cols, rows))
    
    for sub_folder in sub_folder_list:
        if not os.path.isdir(os.path.join(data_root, sub_folder)):
            continue
        if process_seq is not None and sub_folder not in process_seq:
            continue
        print('Processing sequence: {}'.format(sub_folder))

        print('Rendering color and depth images...')
    
        image_output_dir = os.path.join(data_root, sub_folder, "render", "image")
        depth_output_dir = os.path.join(data_root, sub_folder, "render", "depth")
        mask_output_dir = os.path.join(data_root, sub_folder, "render", "mask")

        viewer.reset()
        viewer.scene.floor.enabled = False
        viewer.scene.origin.enabled = False
        viewer.shadows_enabled = False
        # uncomment to turn off the light diffusion
        viewer.scene.lights[0].enabled = False
        viewer.scene.lights[1].enabled = False

        meshes = VariableTopologyMeshes.from_directory(os.path.join(data_root, sub_folder, "meshes_pkl"))
        viewer.scene.add(meshes)

        n_frames = meshes.n_frames

        # Position the camera some distance away from the center of the mesh, about mid-height, looking at the center.
        dist = 6

        # How many virtual cameras to synthesize.
        n_cams = 100
        meshes.current_frame_id = 0
        # Create target and circular positions.
        target = np.mean(meshes.vertices[0], axis=0)
        center = target.copy()
        center[1] += 0.6 * (np.max(meshes.vertices[0][:, 1]) - np.min(meshes.vertices[0][:, 1]))
        positions = circle(center, dist, n_cams)
        
        # Get intrinsics camera in OpenCV format.
        fov = 30.0
        f = 1. / np.tan(np.radians(fov / 2))
        scale = rows / 2.0
        c0 = np.array([cols / 2.0, rows / 2.0])
        cam_intrinsics = np.array([[f * scale, 0., c0[0]], [0., f * scale, c0[1]], [0., 0., 1.]])

        camera = PinholeCamera(positions, target[np.newaxis], cols, rows, fov, viewer=viewer)
        viewer.scene.add(camera)
        viewer.set_temp_camera(camera)
        camera.update_matrices(cols, rows)
        
        d = {'extrinsic': [], 'intrinsic': []}
        d['intrinsic'] = cam_intrinsics
        for i in tqdm.tqdm(range(n_frames)):
            meshes.current_frame_id = i
            camera.current_frame_id = (i+25)%n_cams
            camera.update_matrices(cols, rows)
            
            # Save the camera intrinsics and extrinsics in an OpenCV compatible format.
            view_matrix = camera.get_view_matrix()
            R = view_matrix[:3, :3]
            T = view_matrix[:3, 3:4]
            cam_extrinsics = np.hstack([R, T])

            # The OpenCVCamera class expects extrinsics with Y pointing down, so we flip Y and Z.
            cam_extrinsics[1:, :] *= -1.0
            d["extrinsic"].append(np.concatenate([cam_extrinsics,np.array([[0,0,0,1.]])]))
            
            viewer.save_frame(os.path.join(image_output_dir, "color_{:06d}.png".format(i+1)))
            viewer.save_depth(os.path.join(depth_output_dir, "depth_{:06d}.tiff".format(i+1)))
            viewer.save_mask(os.path.join(mask_output_dir, "mask_{:06d}.png".format(i+1)))

            viewer.run()
        d["extrinsic"] = np.array(d["extrinsic"])
        np.savez(os.path.join(data_root, "cameras.npz"), **d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='/data/X_Humans/00035')
    parser.add_argument("--process_seq", type=str, default=None, help="Process a specific sequence, if None, process all sequences")
    args = parser.parse_args()

    MODES = ['train', 'test']
    for mode in MODES:
        render_color_depth_mask_image(os.path.join(args.data_root, mode), process_seq=args.process_seq)
        
if __name__ == "__main__":
    main()
