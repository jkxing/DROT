


import random
from tqdm.std import tqdm
import numpy as np
import torch
from pytorch3d.renderer import Materials
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.transforms import euler_angles_to_matrix

import sys,os,json,time
sys.path.append(".")
from experiments.common import *
from experiments.logger import Logger
from core.LossFunction import PointLossFunction
from core.NvDiffRastRenderer import NVDiffRastFullRenderer

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

class Furniture:
    def __init__(self) -> None:
        self.meshes=[]
        self.origin_trans = []
        
        self.optim_mesh = []
        self.optim_trans=[]
        pass

    def load_furniture_scene(self, device, task, filepath):
        files = os.listdir(filepath)
        scene={"origin_meshes":[],"optim":[],"sensors":[],"meshes":[],"origin_sensors":[],"material":[]}
        for file in files:
            if not file.endswith(".obj"):
                continue
            fullpath = os.path.join(filepath,file)
            mesh = load_objs_as_meshes([fullpath],device=device)
            trans_x = torch.zeros(1,requires_grad=True)
            trans_z = torch.zeros(1,requires_grad=True)
            rot_y = torch.zeros(1,requires_grad=True)
            optim_trans={}
            optim_trans.update({"translation_x":trans_x})
            optim_trans.update({"translation_z":trans_z})
            optim_trans.update({"rotation_y":rot_y})
            scene["optim"]=scene["optim"]+[trans_x,trans_z,rot_y]
            self.meshes.append({"model":mesh,"optim":True,"trans":{},"optim_trans":optim_trans})
        
        cam = np.loadtxt(os.path.join(filepath,"camera.txt")).tolist()
        poss = []
        cens=[]
        for i in range(0,len(cam),2):
            pos = [cam[i+0][0],cam[i+0][2],-cam[i+0][1]]
            rot = [cam[i+1][0],cam[i+1][2],cam[i+1][1]]
            v = torch.tensor([0,-1,0]).float()
            rot = torch.tensor(rot).float()
            dir = torch.matmul(v,euler_angles_to_matrix(rot,convention="XYZ"))
            dir[2]*=-1
            center = torch.tensor(pos).float()+dir
            poss.append(pos)
            cens.append(center.tolist())

        task["view"].update({"num":len(poss),"direction":"manual","position":poss,"center":cens})
        return scene, task
    
    def gen_mesh(self,optim):
        tmp = []
        for mesh_info in self.meshes:
            src_mesh = mesh_info["model"].clone()
            if optim==True:
                trans_x = mesh_info["optim_trans"]["translation_x"]
                trans_z = mesh_info["optim_trans"]["translation_z"]
                rot_y = mesh_info["optim_trans"]["rotation_y"]
                rot = torch.cat([torch.tensor([0.0]),rot_y,torch.tensor([0.0])]).to(device)
                optim_trans = torch.cat([trans_x,torch.tensor([0.0]),trans_z]).to(device)
                optim_rot = euler_angles_to_matrix(rot,convention="XYZ")
                with torch.no_grad():
                    rot_center = torch.mean(src_mesh.verts_list()[0],axis=0)
                src_mesh.offset_verts_(-rot_center)
                src_mesh.transform_verts_(optim_rot)
                src_mesh.offset_verts_(rot_center)
                src_mesh.offset_verts_(optim_trans)
            tmp.append(src_mesh)
        return join_meshes_as_scene(tmp)
    
if __name__ == "__main__":
    
    argv = sys.argv
    task_name = argv[0].split("/")[1]
    config_path = os.path.join(argv[0].split("/")[0],task_name,sys.argv[1]+".json")
    task = json.load(open(config_path))

    bbox = task["bbox"]
    show = task.get("show", True)
    debug = task.get("debug", False)
    methods = task.get("method", ["origin", "ours"])
    resolution = task["resolution"]
    settings = task["setting"]

    Logger.init(exp_name=task_name, show=show, debug=debug, path="results/")
    #Logger.save_config(task)
    #Logger.save_file("./PointRenderer/NvDiffRastRenderer.py")
    #Logger.save_file("./PointRenderer/LossFunction.py")

    material = Materials(device=device, ambient_color=((0.5, 0.5, 0.5),),specular_color=((0.2,0.2,0.2),),diffuse_color=((0.7,0.7,0.7),))

    testnum=0
    with torch.no_grad():
        model = Furniture()
        gt_scene,task = model.load_furniture_scene(device, task, task["gt_scene_file"])
        sensors, torch_cameras, torch_lights = config_view_light(task, device)
        num_views = len(torch_cameras)
        gt_mesh = model.gen_mesh(optim=False)
        setup_seed(testnum)
        gt_scene["material"]=material
        gt_scene["meshes"]= [{"model": gt_mesh}]
        gt_scene["sensors"]= sensors

        renderer = NVDiffRastFullRenderer(device=device, settings=task.get(
            "renderer"), resolution=resolution)
        
        torch_gt_renderer, torch_sil_renderer, torch_soft_renderer, torch_point_renderer = get_pytorch3d_renderer(
            task.get("renderer").get("background", True), faces_per_pixel=4, resolution=resolution[0], persp=True)
        
        gt_img = renderer.render(gt_scene,DcDt=False)
        for i in range(num_views):
            show_img(gt_img["images"][i],title=str(i),flip=True)

        Logger.save_scene(gt_scene,name="%03d/scene_gt"%(testnum))
        gt_img_torch = [torch_gt_renderer(gt_scene["meshes"][0]["model"], cameras=torch_cameras[i], lights=torch_lights[i], materials = material)[0] for i in range(num_views)]
        gt_sil_torch = [torch_sil_renderer(gt_scene["meshes"][0]["model"], cameras=torch_cameras[i], lights=torch_lights[i], materials = material)[0] for i in range(num_views)]

        for i in range(num_views):
            Logger.save_img("%03d_gtimg_torch_%d.png" % ( 
                testnum,i), gt_img_torch[i].cpu().numpy(), flip=False)
            Logger.save_img("%03d_gtimg_nv_%d.png" % ( 
                testnum,i), gt_img["images"][i].cpu().numpy(), flip=True)
    
    for method in methods:
        setup_seed(testnum)
        model = Furniture()
        scene,task = model.load_furniture_scene(device, task, task["optim_scene_file"])
        sensors, torch_cameras, torch_lights = config_view_light(task, device)
        num_views = len(torch_cameras)
        setup_seed(testnum)
        scene["material"]=material
        scene["sensors"]= sensors
        optimizer_type = settings.get("optimizer","Adam")

        gamma = settings["decay"]
        lr = settings["learning_rate"]
        if settings["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(scene["optim"], lr=lr)
        else:
            optimizer = torch.optim.SGD(
                scene["optim"], lr=lr, momentum=0.9)
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=gamma, last_epoch=-1)
        
        view_per_iter = min(num_views, task["setting"]["view_per_iter"])

        #if method in ["our","finetune"]:
        #    pbar = tqdm(range(settings["Niter"]))
        #elif method[:9]=="pytorch3d":
        #    pbar = tqdm(range(int(settings["Niter"]*2.5)))
        #else:
        #    pbar = tqdm(range(settings["Niter"]*5))

        pbar = tqdm(range(settings["Niter"]))
        loss_func = PointLossFunction(
            debug=debug, 
            resolution=task["resolution"], 
            settings = task["matching"], 
            device=device, 
            renderer=renderer, 
            num_views=num_views,
            logger=Logger)
        
        start_time = time.time()
        for iter in pbar:
            optim_mesh = model.gen_mesh(True)
            scene["meshes"]=[{"model": optim_mesh}]
            if method==methods[0] and iter==0:
                Logger.save_scene(scene,name="%03d/scene_init"%(testnum))
                with torch.no_grad():
                    render_res = renderer.render(scene, DcDt=False)
                    for i in range(num_views):
                        Logger.save_img("%03d/init_img_nv_%d.png" % (testnum,i), render_res["images"][i].cpu().numpy(), flip=True)
                   
            loss_all = torch.tensor(0.0, device=device)
            for j in range(view_per_iter):
                select_view = np.random.randint(0,num_views)
                if method == "our_torch":
                    render_res = torch_point_renderer.render(scene["meshes"][0]["model"], cameras=torch_cameras[select_view], lights=torch_lights[select_view], materials = material)
                    loss = loss_func.get_loss(render_res, gt_img_torch[select_view][...,:3],view=0)
                if method == "our":
                    loss, render_res = loss_func(
                        gt_img, iteration=iter, scene=scene, view=select_view)
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
                loss_all+=loss
                with torch.no_grad():
                    render_res = renderer.render(scene, DcDt=False)
                    for i in range(num_views):
                        Logger.add_image("%03d_render_%s_%d"%(testnum,method,i),render_res["images"][i],flip=True)
                    #cv2.waitKey(0)
            
            optimizer.zero_grad()
            loss_all/=view_per_iter
            loss_all.backward()
            optimizer.step()
            if iter<settings["Niter"]:
                scheduler.step()
            
            MAE = torch.mean(torch.abs(gt_mesh.verts_list()[0]-optim_mesh.verts_list()[0])).item()
            pbar.set_description("%d %s MAE:%.4f" % (testnum, method, MAE))
            Logger.add_scalar("%03d_render_%s_mae" % (testnum, method), MAE, iter)
        
        end_time = time.time()   
        render_res = renderer.render(scene, DcDt=False)
        log_final(scene,gt_img,render_res,testnum,method,Logger,num_views,gt_mesh)
        Logger.clean()
        Logger.show_metric()
    Logger.exit()
