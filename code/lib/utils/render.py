
import numpy as np
import torch
import cv2

from pytorch3d.renderer import (
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointLights
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures


class Renderer():
    def __init__(self, image_size=512):
        super().__init__()

        self.image_size = image_size

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        R = torch.from_numpy(np.array([[-1., 0., 0.],
                                       [0., 1., 0.],
                                       [0., 0., -1.]])).cuda().float().unsqueeze(0)
        t = torch.from_numpy(np.array([[0., 0.3, 5.]])).cuda().float()

        self.cameras = FoVOrthographicCameras(R=R, T=t,device=self.device)

        self.lights = PointLights(device=self.device,location=[[0.0, 0.0, 3.0]],
                            ambient_color=((1,1,1),),diffuse_color=((0,0,0),),specular_color=((0,0,0),))
        self.raster_settings = RasterizationSettings(image_size=image_size,faces_per_pixel=100,blur_radius=0,max_faces_per_bin=60000)
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

        self.shader = HardPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)

        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)
        
    def render_mesh(self, verts, faces, colors=None, mode='npat'):
        '''
        mode: normal, phong, texture
        '''
        with torch.no_grad():

            mesh = Meshes(verts, faces)

            normals = torch.stack(mesh.verts_normals_list())
            front_light = torch.tensor([0,0,1]).float().to(verts.device)
            shades = (normals * front_light.view(1,1,3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)
            results = []

            # normal
            if 'n' in mode:
                normals_vis = normals* 0.5 + 0.5 
                mesh_normal = Meshes(verts, faces, textures=Textures(verts_rgb=normals_vis))
                image_normal = self.renderer(mesh_normal)
                results.append(image_normal)

            # shading
            if 'p' in mode:
                mesh_shading = Meshes(verts, faces, textures=Textures(verts_rgb=shades))
                image_phong = self.renderer(mesh_shading)
                results.append(image_phong)

            # albedo
            if 'a' in mode: 
                assert(colors is not None)
                mesh_albido = Meshes(verts, faces, textures=Textures(verts_rgb=colors))
                image_color = self.renderer(mesh_albido)
                results.append(image_color)
            
            # albedo*shading
            if 't' in mode: 
                assert(colors is not None)
                mesh_teture = Meshes(verts, faces, textures=Textures(verts_rgb=colors*shades))
                image_color = self.renderer(mesh_teture)
                results.append(image_color)

            return  torch.cat(results, axis=1)

image_size = 512
torch.cuda.set_device(torch.device("cuda:0"))
renderer = Renderer(image_size)

def render(verts, faces, colors=None):
    return renderer.render_mesh(verts, faces, colors)

def render_trimesh(mesh, mode='npta'):
    verts = torch.tensor(mesh.vertices).cuda().float()[None]
    faces = torch.tensor(mesh.faces).cuda()[None]
    colors = torch.tensor(mesh.visual.vertex_colors).float().cuda()[None,...,:3]/255
    image = renderer.render_mesh(verts, faces, colors=colors, mode=mode)[0]
    image = (255*image).data.cpu().numpy().astype(np.uint8)
    return image

def render_joint(smpl_jnts, bone_ids):
    marker_sz = 6
    line_wd = 2

    image = np.ones((image_size, image_size,3), dtype=np.uint8)*255 
    smpl_jnts[:,1] += 0.3
    smpl_jnts[:,1] = -smpl_jnts[:,1] 
    smpl_jnts = smpl_jnts[:,:2]*image_size/2 + image_size/2

    for b in bone_ids:
        if b[0]<0 : continue
        joint = smpl_jnts[b[0]]
        cv2.circle(image, joint.astype('int32'), color=(0,0,0), radius=marker_sz, thickness=-1)

        joint2 = smpl_jnts[b[1]]
        cv2.circle(image, joint2.astype('int32'), color=(0,0,0), radius=marker_sz, thickness=-1)

        cv2.line(image, joint2.astype('int32'), joint.astype('int32'), color=(0,0,0), thickness=int(line_wd))

    return image

def weights2colors(weights, model_type):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('Paired')

    if model_type == 'smpl':
        colors = [ 'white', #0
                    'blue', #1
                    'green', #2
                    'red', #3
                    'white', #4
                    'white', #5
                    'white', #6
                    'green', #7
                    'blue', #8
                    'red', #9
                    'white', #10
                    'white', #11
                    'white', #12
                    'blue', #13
                    'green', #14
                    'red', #15
                    'cyan', #16
                    'darkgreen', #17
                    'white', #18
                    'white', #19
                    'blue', #20
                    'green', #21
                    'white', #22
                    'white' #23
        ]
    elif model_type == 'smplh':
        colors = [ 'white', #0
                    'blue', #1
                    'green', #2
                    'red', #3
                    'white', #4
                    'white', #5
                    'white', #6
                    'green', #7
                    'blue', #8
                    'red', #9
                    'white', #10
                    'white', #11
                    'white', #12
                    'blue', #13
                    'green', #14
                    'red', #15
                    'green', #16
                    'blue', #17
                    'blue', #18
                    'green', #19
                    'lightpink', #20
                    'lightpink', #21
                    'red', #22
                    'lightpink', #23
                    'red', #24
                    'green', #25
                    'lightpink', #26
                    'green', #27
                    'darkyellow', #28
                    'lightpink', #29
                    'darkyellow', #30
                    'blue', #31
                    'lightpink', #32
                    'blue', #33
                    'brown', #34
                    'lightpink', #35
                    'brown', #36
                    'red', #37
                    'lightpink', #38
                    'red', #39
                    'green', #40
                    'lightpink', #41
                    'green', #42
                    'darkyellow', #43
                    'lightpink', #44
                    'darkyellow', #45
                    'blue', #46
                    'lightpink', #47
                    'blue', #48
                    'brown', #49
                    'lightpink', #50
                    'brown', #51
        ]
    elif model_type == 'smplx':
        colors = [ 'white', #0
                    'blue', #1
                    'green', #2
                    'red', #3
                    'white', #4
                    'white', #5
                    'white', #6
                    'green', #7
                    'blue', #8
                    'red', #9
                    'white', #10
                    'white', #11
                    'white', #12
                    'blue', #13
                    'green', #14
                    'red', #15
                    'green', #16
                    'blue', #17
                    'blue', #18
                    'green', #19
                    'lightpink', #20
                    'lightpink', #21
                    'brown', #22
                    'darkyellow', #23
                    'green', #24
                    'red', #25
                    'lightpink', #26
                    'red', #27
                    'green', #28
                    'lightpink', #29
                    'green', #30
                    'darkyellow', #31
                    'lightpink', #32
                    'darkyellow', #33
                    'blue', #34
                    'lightpink', #35
                    'blue', #36
                    'brown', #37
                    'lightpink', #38
                    'brown', #39
                    'red', #40
                    'lightpink', #41
                    'red', #42
                    'green', #43
                    'lightpink', #44
                    'green', #45
                    'darkyellow', #46
                    'lightpink', #47
                    'darkyellow', #48
                    'blue', #49
                    'lightpink', #50
                    'blue', #51
                    'brown', #52
                    'lightpink', #53
                    'brown', #54
        ]
    elif model_type == 'mano':
        colors = [ 'lightpink', #0
                    'red', #1
                    'lightpink', #2
                    'red', #3
                    'green', #4
                    'lightpink', #5
                    'green', #6
                    'darkyellow', #7
                    'lightpink', #8
                    'darkyellow', #9
                    'blue', #10
                    'lightpink', #11
                    'blue', #12
                    'brown', #13
                    'lightpink', #14
                    'brown', #15
        ]
    else:
        raise Exception('model_type not supported')

    color_mapping = {'cyan': cmap.colors[3],
                    'blue': cmap.colors[1],
                    'lightpink': cmap.colors[4],
                    'darkgreen': cmap.colors[1],
                    'darkyellow': cmap.colors[7],
                    'brown': cmap.colors[11],
                    'green':cmap.colors[3],
                    'white': [1,1,1],
                    'red':cmap.colors[5],
                    }
    
    for i in range(len(colors)):
        colors[i] = np.array(color_mapping[colors[i]])

    colors = colors[:weights.shape[1]]
    colors = np.stack(colors)[None]
    verts_colors = weights[:,:,None] * colors
    verts_colors = verts_colors.sum(1)
    return verts_colors