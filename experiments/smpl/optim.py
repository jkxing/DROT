
import sys,os,json
sys.path.append(".")
from experiments.common import *

import random
from pytorch3d.renderer import TexturesUV
import numpy as np
from experiments.logger import Logger
from pytorch3d.renderer import Materials
from pytorch3d.structures import Meshes
from core.LossFunction import PointLossFunction
from core.NvDiffRastRenderer import NVDiffRastFullRenderer
from tqdm.std import tqdm
import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

from scipy.io import loadmat
from PIL import Image
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


class SMPL():
    def  __init__(self):
        DATA_DIR = "data/human"
        data_filename = os.path.join(DATA_DIR, "UV_Processed.mat")
        tex_filename = os.path.join(DATA_DIR, "tex.png")
        ALP_UV = loadmat(data_filename)

        with Image.open(tex_filename) as image:
            np_image = np.asarray(image.convert("RGB")).astype(np.float32)
        tex = torch.from_numpy(np_image / 255.)[None].to(device).contiguous()

        verts_temp = torch.from_numpy((ALP_UV["All_vertices"]).astype(
            int)).squeeze().to(device)  # (7829,)
        U = torch.Tensor(ALP_UV['All_U_norm']).to(device)  # (7829, 1)
        V = torch.Tensor(ALP_UV['All_V_norm']).to(device)  # (7829, 1)
        faces = torch.from_numpy(
            (ALP_UV['All_Faces'] - 1).astype(int)).to(device)  # (13774, 3)
        face_indices = torch.Tensor(
            ALP_UV['All_FaceIndices']).squeeze()  # (13774,)
        
        offset_per_part = {}
        already_offset = set()
        cols, rows = 4, 6
        for i, u in enumerate(np.linspace(0, 1, cols, endpoint = False)):
            for j, v in enumerate(np.linspace(0, 1, rows, endpoint = False)):
                part = rows * i + j + 1  # parts are 1-indexed in face_indices
                offset_per_part[part] = (u, v)

        U_norm = U.clone()
        V_norm = V.clone()
        
        # iterate over faces and offset the corresponding vertex u and v values
        for i in range(len(faces)):
            face_vert_idxs = faces[i]
            part = face_indices[i]
            offset_u, offset_v = offset_per_part[int(part.item())]
            for vert_idx in face_vert_idxs:
                if vert_idx.item() not in already_offset:
                    U_norm[vert_idx] = U[vert_idx] / cols + offset_u
                    V_norm[vert_idx] = (1 - V[vert_idx]) / rows + offset_v
                    already_offset.add(vert_idx.item())

        V_norm = 1 - V_norm
        verts_uv = torch.cat([U_norm[None], V_norm[None]], dim = 2).contiguous()  # (1, 7829, 2)
        
        
        smpl_layer = SMPL_Layer(
            center_idx = 0,
            gender = 'male',
            model_root = DATA_DIR)
        
        self.texture = TexturesUV(maps = tex, faces_uvs = faces[None].long().contiguous(), verts_uvs = verts_uv)
        self.smpl_layer = smpl_layer.to(device)
        self.faces = faces
        self.verts_temp=verts_temp

    def gen_mesh(self,pose_params,shape_params):
        verts, Jtr = self.smpl_layer(pose_params, th_betas = shape_params)
        verts = verts[:, self.verts_temp.long()-1]  # (1, 7829, 3)
        mesh = Meshes(verts, self.faces[None], self.texture)
        return mesh
    
if __name__ == "__main__":
    
    argv = sys.argv
    task_name = argv[0].split("/")[1]
    config_path = os.path.join("experiments",task_name,"config.json")
    task = json.load(open(config_path))

    bbox = task["bbox"]
    show = task.get("show", True)
    debug = task.get("debug", False)
    methods = task.get("method", ["origin", "ours"])
    resolution = task["resolution"]
    settings = task["setting"]

    Logger.init(exp_name=task_name, show=show, debug=debug, path="results/")

    material = Materials(device=device, ambient_color=((0.5, 0.5, 0.5),),specular_color=((0.1,0.1,0.1),), diffuse_color=((0.5, 0.5, 0.5),))
    renderer = NVDiffRastFullRenderer(device=device, settings=task.get(
        "renderer"), resolution=resolution)
    torch_gt_renderer, torch_sil_renderer, torch_soft_renderer, torch_point_renderer = get_pytorch3d_renderer(
        task.get("renderer").get("background", True), faces_per_pixel=50, resolution=resolution[0])

    sensors, torch_cameras, torch_lights = config_view_light(task, device)
    num_views = len(torch_cameras)

    scene = {"origin_meshes":[], "meshes": [], "sensors": sensors}
    scene["meshes"].append({})
    scene["material"]=material
    model = SMPL()

    for testnum in range(0,1):
        setup_seed(testnum)
        
        pose_params = (torch.rand(1, 72) - 0.5) * 0.8
        shape_params = torch.rand(1, 10) * 0.0
        position = (torch.rand(3) - 0.5) * 0.8

        pose_params = pose_params.to(device)
        shape_params = shape_params.to(device)
        position = position.to(device)
        with torch.no_grad():
            gt_mesh = model.gen_mesh(pose_params,shape_params)
            gt_mesh.material = material
            scene["meshes"][0] = {"model": gt_mesh}
            scene["origin_meshes"]=[{"init_model": gt_mesh}]
            Logger.save_scene(scene, name="%03d/scene_gt"%(testnum))
            gt_img = renderer.render(scene,DcDt=False)
            gt_img_torch = [torch_gt_renderer(scene["meshes"][0]["model"], cameras=torch_cameras[i], lights=torch_lights[i], materials = material)[0] for i in range(num_views)]
            gt_img_torch_soft = [torch_soft_renderer(scene["meshes"][0]["model"], cameras=torch_cameras[i], lights=torch_lights[i], materials = material)[0] for i in range(num_views)]
            gt_sil_torch = [torch_sil_renderer(scene["meshes"][0]["model"], cameras=torch_cameras[i], lights=torch_lights[i], materials = material)[0] for i in range(num_views)]
            for i in range(num_views):
                Logger.save_img("%03d/img_gt_nv_%d.png" %(testnum,i), gt_img["images"][i].cpu().numpy(), flip=True)
            for i in range(num_views):
                Logger.save_img("%03d/img_gt_torch_%d.png" %(testnum,i), gt_img_torch[i].cpu().numpy(), flip=False)
            for i in range(num_views):
                Logger.save_img("%03d/img_gt_torch_soft_%d.png" %(testnum,i), gt_img_torch_soft[i].cpu().numpy(), flip=False)
            
        Logger.save_txt("%03d/pose_gt.txt"%(testnum), pose_params.cpu().numpy())
        Logger.save_npy("%03d/img_gt.npy" %(testnum), gt_img["images"].cpu().numpy())
        #Logger.save_img("%03d/img_gt.png" %(testnum), gt_img["images"].cpu().numpy()[0], flip=True)
        #Logger.add_image(name="%03d_gtimg_torch" % (
        #    testnum), content=gt_img_torch[0], flip=False, type="image", step=0)
        #Logger.add_image(name="%03d_gtimg_nv" % (
        #    testnum), content=gt_img["images"][0], flip=True, type="image", step=0)
        #cv2.waitKey(0)
        #continue
        for method in methods:

            pose_params_opt = torch.zeros((1, 72), device = device, requires_grad = True)
            shape_params_opt = torch.zeros((1, 10), device = device, requires_grad = True)   
            position = torch.zeros(3, device = device, requires_grad = True)
            
            scene["optim"] = [pose_params_opt]
            loss_func = PointLossFunction(
                debug=debug,
                resolution=task["resolution"],
                settings=task["matching"],
                device=device,
                renderer=renderer,
                num_views=task["view"]["num"],
                logger=Logger)
            
            gamma = settings["decay"]
            lr = settings["learning_rate"]
            if settings["optimizer"] == "Adam":
                optimizer = torch.optim.Adam([pose_params_opt], lr=lr)
            else:
                optimizer = torch.optim.SGD([pose_params_opt], lr=lr, momentum=0.9)
            
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma, last_epoch=-1)
            
            view_per_iter = min(num_views, task["setting"]["view_per_iter"])
            if method in ["our","finetune"]:
                pbar = tqdm(range(settings["Niter"]))
            elif method[:9]=="pytorch3d":
                pbar = tqdm(range(int(settings["Niter"]*1.2)))
            else:
                pbar = tqdm(range(settings["Niter"]*2))
            
            for iter in pbar:
                optim_mesh = model.gen_mesh(pose_params_opt, shape_params_opt)
                optim_mesh.material = material
                scene["meshes"][0] = {"model": optim_mesh}
                loss_all = torch.tensor(0.0, device=device)
                
                for j in range(view_per_iter):
                    select_view = np.random.randint(0,num_views)
                    if method == "our":
                        loss, render_res = loss_func(gt_img, iteration=iter, scene=scene, view=select_view)
                    elif method == "nvdiffrast":
                        render_res = renderer.render(scene, DcDt=True, view=select_view)
                        loss = torch.mean((render_res["images"][0]-gt_img["images"][select_view])**2)
                    elif method == "random":
                        x = random.randint(0, 1)
                        if x == 0:
                            loss, render_res = loss_func(gt_img, iteration=iter, scene=scene, view=select_view)
                        else:
                            render_res = renderer.render(scene, DcDt=True, view=select_view)
                            loss = torch.mean((render_res["images"][0]-gt_img["images"][select_view])**2)
                    elif method == "finetune":
                        if iter < settings["Niter"]*0.75:
                            loss, render_res = loss_func(gt_img, iteration=iter, scene=scene, view=select_view)
                        else:
                            if iter==int(settings["Niter"]*0.75):
                                optimizer = torch.optim.SGD(scene["optim"], lr=lr*(gamma**iter))
                            render_res = renderer.render(scene, DcDt=True, view=select_view)
                            loss = torch.mean((render_res["images"][0]-gt_img["images"][select_view])**2)
                            #render_res = torch_soft_renderer(scene["meshes"][0]["model"], cameras=torch_cameras[select_view], lights=torch_lights[select_view], materials = material)
                            #loss = torch.mean((render_res[0,...,3]-gt_sil_torch[select_view][...,3])**2)
                    elif method=="pytorch3d_rgb":
                        render_res = torch_soft_renderer(scene["meshes"][0]["model"], cameras=torch_cameras[select_view], lights=torch_lights[select_view], materials = material)
                        loss = torch.mean((render_res[0,...,:3]-gt_img_torch[select_view][...,:3])**2)
                    elif method=="pytorch3d_sil_rgb":
                        render_res = torch_soft_renderer(scene["meshes"][0]["model"], cameras=torch_cameras[select_view], lights=torch_lights[select_view], materials = material)
                        loss = torch.mean((render_res[0,...,:3]-gt_img_torch[select_view][...,:3])**2)
                        loss += torch.mean((render_res[0,...,3]-gt_sil_torch[select_view][...,3])**2)
                    elif method=="pytorch3d_sil":
                        render_res = torch_soft_renderer(scene["meshes"][0]["model"], cameras=torch_cameras[select_view], lights=torch_lights[select_view], materials = material)
                        loss = torch.mean((render_res[0,...,3]-gt_sil_torch[select_view][...,3])**2)
                    elif method=="our_sil":
                        loss, render_res = loss_func(gt_img, iteration=iter, scene=scene, view=select_view)
                        render_res = torch_soft_renderer(scene["meshes"][0]["model"], cameras=torch_cameras[select_view], lights=torch_lights[select_view], materials = material)
                        loss += torch.mean((render_res[0,...,3]-gt_sil_torch[select_view][...,3])**2)
                    loss_all+=loss

                optimizer.zero_grad()
                loss_all/=view_per_iter
                loss_all.backward()
                optimizer.step()
                if iter<settings["Niter"]:
                    scheduler.step()
                
                mae = torch.mean(torch.abs(optim_mesh.verts_list()[0]-gt_mesh.verts_list()[0])).item()
                Logger.add_scalar("%03d_render_%s_mae" %(testnum, method), mae, iter)
                pbar.set_description("%d %s mae:%.4f" % (testnum, method, mae))
            
            render_res = renderer.render(scene, DcDt=False)
            log_final(scene, gt_img, render_res, testnum, method, Logger, num_views)
            
        Logger.clean()
        Logger.show_metric()
    Logger.exit()
        # cv2.waitKey(0)
