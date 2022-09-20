import torch
import numpy as np
from torch.nn import Module
from geomloss import SamplesLoss
#from sklearn.neighbors import NearestNeighbors

class PointLossFunction(Module):
    def __init__(self, resolution, renderer, device, settings, debug, num_views, logger):
        super().__init__()
        self.num_views = num_views
        
        self.match_weight = settings.get("matching_weight",1.0)
        self.matchings_record=[0 for i in range(num_views)]
        self.matchings = [[] for i in range(num_views)]
        self.rasts = [[] for i in range(num_views)]
        self.rgb_weight = [self.match_weight for i in range(num_views)]
        self.matching_interval = settings.get("matching_interval",0)
        self.renderer = renderer
        self.device = device
        self.resolution = resolution[0]
        self.debug=debug
        self.logger = logger
        self.step = -1
        #Matcher setting
        self.matcher_type=settings.get("matcher","Sinkhorn")
        self.matcher = None
        self.loss = SamplesLoss("sinkhorn", blur=0.01)

        #normal image grid, used for pixel position completion
        x = torch.linspace(0, 1, self.resolution)
        y = torch.linspace(0, 1, self.resolution)
        pos = torch.meshgrid(x, y)
        self.pos = torch.cat([pos[1][..., None], pos[0][..., None]], dim=2).to(self.device)[None,...].repeat(num_views,1,1,1)
        self.pos_np = self.pos[0].clone().cpu().numpy().reshape(-1,2)
    
    def visualize_point(self, res, title, view):#(N,5) (r,g,b,x,y)
        res = res.detach().cpu().numpy()
        X = res[...,3:]
        #need install sklearn
        nbrs = None#NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(self.pos_np)
        distances = np.exp(-distances*self.resolution)
        img = np.sum(res[indices,:3]*distances[...,None],axis = 1)
        img = img/np.sum(distances,axis = 1)[...,None]
        img = img.reshape(self.resolution, self.resolution, 3)
        self.logger.add_image(title+"_"+str(view), img, self.step)

    #unused currently
    def rgb_match_weight(self, view=0):
        return self.rgb_weight[view]

    def match_Sinkhorn(self, haspos, render_point_5d, gt_rgb, view):
        h,w = render_point_5d.shape[1:3]
        target_point_5d = torch.zeros((haspos.shape[0], h, w, 5), device=self.device)
        target_point_5d[..., :3] = torch.clamp(gt_rgb,0,1)
        target_point_5d[..., 3:] = render_point_5d[...,3:].clone().detach()
        target_point_5d = target_point_5d.reshape(-1, h*w, 5)
        render_point_5d_match = render_point_5d.clone().reshape(-1,h*w,5)
        render_point_5d_match.clamp_(0.0,1.0)
        render_point_5d_match[...,:3] *= self.rgb_match_weight(view)
        target_point_5d[...,:3] = target_point_5d[...,:3]*self.rgb_match_weight(view)
        pointloss = self.loss(render_point_5d_match, target_point_5d)*self.resolution*self.resolution
        [g] = torch.autograd.grad(torch.sum(pointloss), [render_point_5d_match])
        g[...,:3]/=self.rgb_match_weight(view)
        return (render_point_5d-g.reshape(-1,h,w,5)).detach()
    
    def get_loss(self, render_res, gt_rgb, view):
        haspos = render_res["msk"]
        render_pos = (render_res["pos"]+1.0)/2.0
        render_rgb = render_res["images"][...,:3]
        render_pos[haspos==False]=self.pos[view:view+1][haspos==False].clone()
        render_point_5d = torch.cat([render_rgb, render_pos], dim=-1)
        match_point_5d = self.match_Sinkhorn(haspos, render_point_5d, gt_rgb, view)
        disp = match_point_5d-render_point_5d
        loss = torch.mean(disp**2)
        return loss
    
    def forward(self, gt, iteration=-1, scene=None, view=0):
        self.step=iteration

        new_match = ((self.matchings_record[view] % (self.matching_interval+1))==0)

        if new_match:
            render_res = self.renderer.render(scene, view=view, DcDt=False)
            self.rasts[view] = render_res["rasts"]
        else:
            render_res = self.renderer.render(scene, rasts_list = self.rasts[view], view=view, DcDt=False)
        
        self.matchings_record[view] += 1
        haspos = render_res["msk"]
        render_pos = (render_res["pos"]+1.0)/2.0
        render_rgb = render_res["images"]
        render_pos[haspos==False]=self.pos[view:view+1][haspos==False].clone()
        render_point_5d = torch.cat([render_rgb, render_pos], dim=-1)
        gt_rgb=gt["images"][view:view+1]
        if new_match:
            if self.matcher_type=="Sinkhorn":
                self.matchings[view] = self.match_Sinkhorn(haspos, render_point_5d, gt_rgb, view)

        match_point_5d = self.matchings[view]
        disp = match_point_5d-render_point_5d
        loss = torch.mean(disp**2)

        if self.debug:
            self.visualize_point(match_point_5d.reshape(-1,5),title="match",view=view)
        
        return loss, render_res
            

            

