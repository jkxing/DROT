

import sys,os,json
sys.path.append(".")
from experiments.common import *

import random,time
from pytorch3d.structures import join_meshes_as_scene
import numpy as np
from experiments.logger import Logger
from pytorch3d.renderer import Materials
from core.LossFunction import PointLossFunction
from core.NvDiffRastRenderer import NVDiffRastFullRenderer
from tqdm.std import tqdm
import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


if __name__ == "__main__":

    argv = sys.argv
    task_name = argv[0].split("/")[1]
    config_path = os.path.join("experiments",task_name,argv[1]+".json")

    task = json.load(open(config_path))

    bbox = task.get("bbox", 1.0)
    testnums = task.get("testnums", 10)
    show = task.get("show", True)
    debug = task.get("debug", False)
    methods = task.get("method", ["origin", "ours"])
    resolution = task["resolution"]
    settings = task["setting"]

    Logger.init(exp_name=argv[1], show=show, debug=debug, path="results/")

    Logger.save_config(task)
    #Logger.save_file("./core/NvDiffRastRenderer.py")
    #Logger.save_file("./core/LossFunction.py")
    #Logger.save_file(argv[0])
    
    renderer = NVDiffRastFullRenderer(device=device, settings=task.get(
        "renderer"), resolution=resolution)
    torch_gt_renderer, torch_sil_renderer, torch_soft_renderer, torch_point_renderer = get_pytorch3d_renderer(
        task.get("renderer").get("background", True), faces_per_pixel=50, resolution=resolution[0],sigma=1e-5)
    
    for testnum in range(testnums):  
        setup_seed(testnum) 
        sensors, torch_cameras, torch_lights = config_view_light(task, device)
        num_views = len(torch_cameras)

        meshes = load_scene(task["scene_file"], device)
        gt_mesh = join_meshes_as_scene(meshes)
        
        material = Materials(device=device, ambient_color=((0.0, 0.0, 0.0),),diffuse_color=((0.8,0.8,0.8),), specular_color=((0.2, 0.2, 0.2),)).to(device)
        
        origin_scene = {"origin_meshes": [], "origin_sensors": sensors, "material":material}
        for mesh in meshes:
            origin_scene["origin_meshes"].append({"model":mesh})

        with torch.no_grad():
            gt_scene = config_scene(origin_scene, task, optim=False)
            gt_scene = get_current_scene(gt_scene, device=device)
            Logger.save_scene(gt_scene, name="%03d/scene_gt"%(testnum))
            gt_img = renderer.render(gt_scene,DcDt=False)
            for view in range(num_views):
                Logger.save_img("%03d/img_gt_%d.png" %(testnum,view), gt_img["images"][view].cpu().numpy())
        
        for method in methods:
            setup_seed(testnum) 
            meshes = load_scene(task["scene_file"], device)
            gt_mesh = join_meshes_as_scene(meshes)
            
            material = Materials(device=device, ambient_color=((0.0, 0.0, 0.0),),diffuse_color=((0.8,0.8,0.8),), specular_color=((0.2, 0.2, 0.2),)).to(device)
            
            origin_scene = {"origin_meshes": [], "origin_sensors": sensors, "material":material}
            for mesh in meshes:
                origin_scene["origin_meshes"].append({"model":mesh})
            init_scene = config_scene(origin_scene,task,device=device)
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
                optimizer = torch.optim.Adam(init_scene["optim"], lr=lr)
            else:
                optimizer = torch.optim.SGD(init_scene["optim"], lr=lr, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma, last_epoch=-1)

            view_per_iter = min(num_views, task["setting"]["view_per_iter"])
            if method==methods[0]:
                scene = get_current_scene(init_scene, device=device)
                Logger.save_scene(scene,name="%03d/scene_init"%(testnum))
                with torch.no_grad():
                    render_res = renderer.render(scene, DcDt=False)
                    for view in range(num_views):
                        Logger.save_img("%03d/img_init_%d.png" %(testnum,view), render_res["images"][view].cpu().numpy())
            
            pbar = tqdm(range(settings["Niter"]))
            start_time = time.time()
            for iter in pbar:
                scene = get_current_scene(init_scene, device=device)
                loss_all = torch.tensor(0.0,device=device)
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
                            render_res = renderer.render(scene, DcDt=True, view=select_view)
                            loss = torch.mean((render_res["images"][0]-gt_img["images"][select_view])**2)
                    loss_all+=loss
                loss_all/=view_per_iter
                optimizer.zero_grad()
                loss_all.backward()
                optimizer.step()
                scheduler.step()
                
                mae = torch.mean(torch.abs(scene["meshes"][0]["model"].verts_list()[0]-gt_mesh.verts_list()[0])).item()
                pbar.set_description("%d %s mae:%.4f" % (testnum, method, mae))
                #Logger.add_scalar("render_mae",mae,iter)
                #Logger.step(stop=True)
                
            end_time = time.time()
            Logger.add_metric("metric_speed_%s"%(method), settings["Niter"]/(end_time-start_time))
            with torch.no_grad():
                render_res = renderer.render(scene, DcDt=False)
                log_final(scene, gt_img, render_res, testnum, method, Logger, num_views, gt_mesh)
            Logger.clean()
        Logger.show_metric()
    Logger.exit()
