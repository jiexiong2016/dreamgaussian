import os
import tqdm

import numpy as np
import torch

from cam_utils import orbit_camera, perspective
from grid_put import mipmap_linear_grid_put_2d
from gs_renderer import MiniCam, Renderer
from guidance.zero123_utils import Zero123
from mesh import safe_normalize
from mesh_renderer import Renderer as MeshRenderer

from omegaconf import OmegaConf

class DreamGaussianModel():

    def __init__(self, config_file="./configs/dream_gsplat.yaml"):
        assert torch.cuda.is_available()
        self.device = torch.device("cuda")

        self.dg_opt = OmegaConf.load(config_file)
        self.train_steps = 500
        self.refine_steps = 50
        self.guidance = Zero123(self.device)

        self.gaussian_render = None
        self.gaussian_optimizer = None

        self.mesh = None
        self.mesh_renderer = None
        self.mesh_optimizer = None

    def create_mini_cam(self, pose, render_resolution):
        return MiniCam(
            pose,
            render_resolution,
            render_resolution,
            np.deg2rad(self.dg_opt.fovy),
            np.deg2rad(self.dg_opt.fovy),
            self.dg_opt.z_near,
            self.dg_opt.z_far
        )

    def calculate_reference_loss(self, fixed_view, input_image, input_alpha, step_ratio):
        image = fixed_view["image"].unsqueeze(0)
        loss = 10000 * step_ratio * torch.nn.functional.mse_loss(image, input_image)

        alpha = fixed_view["alpha"].unsqueeze(0)
        loss += 1000 * step_ratio * torch.nn.functional.mse_loss(alpha, input_alpha)
        return loss

    def densify_and_prune(self, step, fixed_view):
        if step >= self.dg_opt.density_start_iter and step <= self.dg_opt.density_end_iter:
            viewspace_point_tensor, visibility_filter, radii = fixed_view["viewspace_points"], fixed_view["visibility_filter"], fixed_view["radii"]
            self.gaussian_render.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussian_render.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            self.gaussian_render.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if step % self.dg_opt.densification_interval == 0:
                self.gaussian_render.gaussians.densify_and_prune(self.dg_opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
            
            if step % self.dg_opt.opacity_reset_interval == 0:
                self.gaussian_render.gaussians.reset_opacity()

    def train(self, input_image, input_alpha):

        self.gaussian_render = Renderer(sh_degree=0)
        self.gaussian_render.initialize(num_pts=self.dg_opt.num_pts)

        self.gaussian_render.gaussians.training_setup(self.dg_opt)
        self.gaussian_render.gaussians.active_sh_degree = self.gaussian_render.gaussians.max_sh_degree
        self.gaussian_optimizer = self.gaussian_render.gaussians.optimizer

        with torch.no_grad():
            self.guidance.get_img_embeds(input_image)

        pose = orbit_camera(self.dg_opt.elevation, 0, self.dg_opt.radius)
        fixed_cam = self.create_mini_cam(pose, self.dg_opt.ref_size)
    
        min_ver = max(min(-30, -30 - self.dg_opt.elevation), -80 - self.dg_opt.elevation)
        max_ver = min(max(30, 30 - self.dg_opt.elevation), 80 - self.dg_opt.elevation)

        for step in tqdm.trange(self.train_steps):

            cur_step = step + 1
            step_ratio = cur_step / self.train_steps
            self.gaussian_render.gaussians.update_learning_rate(cur_step)
            
            fixed_view = self.gaussian_render.render(fixed_cam)
            loss = self.calculate_reference_loss(fixed_view, input_image, input_alpha, step_ratio)

            render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)

            ver = np.random.randint(min_ver, max_ver)
            hor = np.random.randint(-180, 180)

            random_pose = orbit_camera(self.dg_opt.elevation + ver, hor, self.dg_opt.radius)
            random_cam = self.create_mini_cam(random_pose, render_resolution)

            bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.dg_opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
            random_view = self.gaussian_render.render(random_cam, bg_color=bg_color)
            random_view_image = random_view["image"].unsqueeze(0)

            loss += self.guidance.train_step(random_view_image, [ver], [hor], [0], step_ratio=step_ratio)

            loss.backward()
            self.gaussian_optimizer.step()
            self.gaussian_optimizer.zero_grad()

            self.densify_and_prune(cur_step, fixed_view)
        self.extract_mesh()

    def refine(self, input_image):

        self.mesh_renderer = MeshRenderer(self.dg_opt, mesh=self.mesh).to(self.device)
        self.mesh_optimizer = torch.optim.Adam(self.mesh_renderer.get_params())

        with torch.no_grad():
            self.guidance.get_img_embeds(input_image)

        input_image_channel_last = input_image[0].permute(1,2,0).contiguous()

        pose = orbit_camera(self.dg_opt.elevation, 0, self.dg_opt.radius)
        perspect = perspective(np.deg2rad(self.dg_opt.fovy), self.dg_opt.ref_size, self.dg_opt.ref_size, self.dg_opt.z_near, self.dg_opt.z_far)

        min_ver = max(min(-30, -30 - self.dg_opt.elevation), -80 - self.dg_opt.elevation)
        max_ver = min(max(30, 30 - self.dg_opt.elevation), 80 - self.dg_opt.elevation)

        for step in tqdm.trange(self.refine_steps):
            cur_step = step + 1
            step_ratio = min(1, cur_step / self.refine_steps)

            ssaa = min(2.0, max(0.125, 2 * np.random.random()))
            fixed_view = self.mesh_renderer.render(pose, perspect, self.dg_opt.ref_size, self.dg_opt.ref_size, ssaa=ssaa)

            fixed_view_image = fixed_view["image"] # [H, W, 3] in [0, 1]
            fixed_view_mask = ((fixed_view["alpha"] > 0) & (fixed_view["viewcos"] > 0.5)).detach()

            loss = torch.nn.functional.mse_loss(fixed_view_image * fixed_view_mask, input_image_channel_last * fixed_view_mask)
            
            ver = np.random.randint(min_ver, max_ver)
            hor = np.random.randint(-180, 180)
            
            render_resolution = 512
            random_pose = orbit_camera(self.dg_opt.elevation + ver, hor, self.dg_opt.radius)

            # random render resolution
            ssaa = min(2.0, max(0.125, 2 * np.random.random()))
            random_view = self.mesh_renderer.render(random_pose, perspect, render_resolution, render_resolution, ssaa=ssaa)
            random_view_image = random_view["image"].permute(2,0,1).contiguous().unsqueeze(0)

            strength = step_ratio * 0.15 + 0.8
            refined_images = self.guidance.refine(random_view_image, [ver], [hor], [0], strength=strength).float()
            refined_images = torch.nn.functional.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
            loss = loss + torch.nn.functional.mse_loss(random_view_image, refined_images)
                
            loss.backward()
            self.mesh_optimizer.step()
            self.mesh_optimizer.zero_grad()

        return self.mesh_renderer.mesh.pack()

    @torch.no_grad()
    def extract_mesh(self, texture_size=1024):
        mesh = self.gaussian_render.gaussians.extract_mesh(density_thresh=self.dg_opt.density_thresh)

        print(f"[INFO] unwrap uv...")
        h = w = texture_size
        mesh.auto_uv()
        mesh.auto_normal()

        albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
        cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

        vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
        hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

        render_resolution = 512

        import nvdiffrast.torch as dr
        glctx = dr.RasterizeCudaContext()

        for ver, hor in zip(vers, hors):
            # render image
            pose = orbit_camera(ver, hor, self.dg_opt.radius)
            cur_cam = self.create_mini_cam(pose, render_resolution)
            cur_out = self.gaussian_render.render(cur_cam)

            rgbs = cur_out["image"].unsqueeze(0)

            # get coordinate in texture image
            pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)

            perspect = perspective(np.deg2rad(self.dg_opt.fovy), render_resolution, render_resolution, self.dg_opt.z_near, self.dg_opt.z_far)
            proj = torch.from_numpy(perspect.astype(np.float32)).to(self.device)

            v_cam = torch.matmul(torch.nn.functional.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
            v_clip = v_cam @ proj.T
            rast, _ = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

            depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
            depth = depth.squeeze(0) # [H, W, 1]

            alpha = (rast[0, ..., 3:] > 0).float()

            uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

            # use normal to produce a back-project mask
            normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
            normal = safe_normalize(normal[0])

            # rotated normal (where [0, 0, 1] always faces camera)
            rot_normal = normal @ pose[:3, :3]
            viewcos = rot_normal[..., [2]]

            mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
            mask = mask.view(-1)

            uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
            rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
            
            # update texture image
            cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                h, w,
                uvs[..., [1, 0]] * 2 - 1,
                rgbs,
                min_resolution=256,
                return_count=True,
            )
            
            mask = cnt.squeeze(-1) < 0.1
            albedo[mask] += cur_albedo[mask]
            cnt[mask] += cur_cnt[mask]

        mask = cnt.squeeze(-1) > 0
        albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

        mask = mask.view(h, w)

        albedo = albedo.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        # dilate texture
        from sklearn.neighbors import NearestNeighbors
        from scipy.ndimage import binary_dilation, binary_erosion

        inpaint_region = binary_dilation(mask, iterations=32)
        inpaint_region[mask] = 0

        search_region = mask.copy()
        not_search_region = binary_erosion(search_region, iterations=3)
        search_region[not_search_region] = 0

        search_coords = np.stack(np.nonzero(search_region), axis=-1)
        inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

        knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
            search_coords
        )
        _, indices = knn.kneighbors(inpaint_coords)

        albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]
        mesh.albedo = torch.from_numpy(albedo).to(self.device)
        self.mesh = mesh
