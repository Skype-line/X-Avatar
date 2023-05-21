
# Dataset Description:

X-Humans Dataset contains high-quality textured human scans, rendered RGB-D images, and SMPL[-X] registrations. Totally, it has 20 subjects, 233 sequences, 35,427 frames. It includes individuals with a variety of body postures, hand gestures, and facial expressions, as well as diverse clothing styles, hairstyles, genders, and ages.

Name/ID | Gender | #Seq (Train) | #Seq (Test) | #Frames (Train) | #Frames (Test)
--- | --- | --- | --- |--- |--- |
00016 |  male |  9 |  2 |  1330 |  296
00017 |  male |  9 |  2 |  1339 |  299
00018 |  male |  9 |  2 |  1340 |  300
00019 |  female |  7 |  2 |  1015 |  273
00020 |  male |  10 |  3 |  1721 |  376
00021 |  female |  6 |  1 |  1000 |  150
00024 |  male |  12 |  3 |  1799 |  394
00025 |  female |  9 |  2 |  1335 |  300
00027 |  female |  10 |  2 |  1439 |  300
00028 |  male |  10 |  3 |  1444 |  448
00034 |  male |  10 |  2 |  1411 |  300
00035 |  male |  10 |  2 |  1322 |  300
00036 |  female |  10 |  3 |  1464 |  449
00039 |  female |  10 |  2 |  1384 |  292
00041 |  male |  12 |  2 |  1727 |  300
00058 |  female |  10 |  2 |  1426 |  295
00085 |  female |  8 |  2 |  1595 |  330
00086 |  female |  10 |  2 |  1738 |  360
00087 |  male |  9 |  2 |  1780 |  360
00088 |  male |  10 |  2 |  1396 |  300
Total |  / | 190 |  43 |  29005 |  6422

# File structure:

    
    X_Humans
    └── subject ID/
        ├── gender.txt: gender
        ├── mean_shape_smpl.npy: SMPL 'betas' (10,)
        ├── mean_shape_smplx.npy: SMPL-X 'betas' (10,)
        └── mode(train/test)/
            └── Sequence ID/
                ├── meshes_pkl/
                │   ├── atlas-fxxxxx.pkl: low-res textures as pickle files (1024, 1024, 3)
                │   └── mesh-fxxxxx.pkl: 'vertices', 'normals', 'uvs', 'faces'
                ├── render/
                │   ├── depth/
                │   │   └── depth_xxxxxx.tiff, depth image
                │   ├── image/
                │   │   └── color_xxxxxx.png, RGB image
                │   └── cameras.npz: intrinsic and extrinsic of the virtual camera
                ├── SMPL/
                │   ├── mesh-fxxxxx_smpl.pkl: SMPL params 'global_orient' (3,), 'transl' (3,), 'body_pose' (69,), 'betas' (10,) (use mean_shape_smpl.npy instead)
                │   └── mesh-fxxxxx_smpl.ply: SMPL meshes
                └── SMPLX/
                    ├── mesh-fxxxxx_smplx.pkl: SMPL-X params 'global_orient' (3,), 'transl' (3,), 'body_pose' (63,), 'left_hand_pose' (45,), 'right_hand_pose' (45,), 
                    │                         'jaw_pose' (3,), 'leye_pose' (3,), 'reye_pose' (3,), 'expression' (10,), 'betas' (10,) (use mean_shape_smplx.npy instead)
                    └── mesh-fxxxxx_smplx.ply: SMPL-X meshes

# Code instruction:
- preprocess_XHumans.py: prepare X-Humans dataset for training X-Avatar https://github.com/Skype-line/X-Avatar
- vis_meshes.py: visualize X-Humans 3D scans and SMPL[-X] meshes with aitviewer https://github.com/eth-ait/aitviewer