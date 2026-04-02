import argparse
import gradio
import os
import torch
import numpy as np
import tempfile
import functools
import subprocess
import matplotlib.pyplot as plt

from slam3r.pipeline.recon_offline_pipeline import get_img_tokens, initialize_scene, adapt_keyframe_stride, i2p_inference_batch, l2w_inference, normalize_views, scene_frame_retrieve
from slam3r.datasets.wild_seq import Seq_Data
from slam3r.models import Local2WorldModel, Image2PointsModel
from slam3r.utils.device import to_numpy
from slam3r.utils.recon_utils import *
from slam3r.utils.geometry import inv

def extract_frames(video_path: str, fps: float) -> str:
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "%03d.jpg")
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        output_path
    ]
    subprocess.run(command, check=True)
    return temp_dir

def recon_scene(i2p_model:Image2PointsModel,
                l2w_model:Local2WorldModel,
                device, save_dir, fps,
                img_dir_or_list,
                keyframe_stride, win_r, initial_winsize, conf_thres_i2p,
                num_scene_frame, update_buffer_intv, buffer_strategy, buffer_size,
                conf_thres_l2w, num_points_save, save_traj=False):
    # print(f"device: {device},\n save_dir: {save_dir},\n fps: {fps},\n keyframe_stride: {keyframe_stride},\n win_r: {win_r},\n initial_winsize: {initial_winsize},\n conf_thres_i2p: {conf_thres_i2p},\n num_scene_frame: {num_scene_frame},\n update_buffer_intv: {update_buffer_intv},\n buffer_strategy: {buffer_strategy},\n buffer_size: {buffer_size},\n conf_thres_l2w: {conf_thres_l2w},\n num_points_save: {num_points_save}")
    np.random.seed(42)
    
    # load the imgs or video
    if isinstance(img_dir_or_list, str):
        img_dir_or_list = extract_frames(img_dir_or_list, fps)
    
    dataset = Seq_Data(img_dir_or_list, to_tensor=True)
    data_views = dataset[0][:]
    num_views = len(data_views)
    
    # Pre-save the RGB images along with their corresponding masks 
    # in preparation for visualization at last.
    rgb_imgs = []
    for i in range(len(data_views)):
        if data_views[i]['img'].shape[0] == 1:
            data_views[i]['img'] = data_views[i]['img'][0]        
        rgb_imgs.append(transform_img(dict(img=data_views[i]['img'][None]))[...,::-1])
    
    #preprocess data for extracting their img tokens with encoder
    for view in data_views:
        view['img'] = torch.tensor(view['img'][None])
        view['true_shape'] = torch.tensor(view['true_shape'][None])
        for key in ['valid_mask', 'pts3d_cam', 'pts3d']:
            if key in view:
                del view[key]
        to_device(view, device=device)
    # pre-extract img tokens by encoder, which can be reused 
    # in the following inference by both i2p and l2w models
    res_shapes, res_feats, res_poses = get_img_tokens(data_views, i2p_model)    # 300+fps
    print('finish pre-extracting img tokens')

    # re-organize input views for the following inference.
    # Keep necessary attributes only.
    input_views = []
    for i in range(num_views):
        input_views.append(dict(label=data_views[i]['label'],
                              img_tokens=res_feats[i], 
                              true_shape=data_views[i]['true_shape'], 
                              img_pos=res_poses[i]))
    
    # decide the stride of sampling keyframes, as well as other related parameters
    if keyframe_stride == -1:
        kf_stride = adapt_keyframe_stride(input_views, i2p_model, 
                                          win_r = 3,
                                          adapt_min=1,
                                          adapt_max=20,
                                          adapt_stride=1)
    else:
        kf_stride = keyframe_stride
    
    # initialize the scene with the first several frames
    initial_winsize = min(initial_winsize, num_views//kf_stride)
    assert initial_winsize >= 2, "not enough views for initializing the scene reconstruction"
    initial_pcds, initial_confs, init_ref_id = initialize_scene(input_views[:initial_winsize*kf_stride:kf_stride], 
                                                   i2p_model, 
                                                   winsize=initial_winsize,
                                                   return_ref_id=True) # 5*(1,224,224,3)
    
    # start reconstrution of the whole scene
    init_num = len(initial_pcds)
    per_frame_res = dict(i2p_pcds=[], i2p_confs=[], l2w_pcds=[], l2w_confs=[])
    for key in per_frame_res:
        per_frame_res[key] = [None for _ in range(num_views)]
    
    registered_confs_mean = [_ for _ in range(num_views)]
    
    # set up the world coordinates with the initial window
    for i in range(init_num):
        per_frame_res['l2w_confs'][i*kf_stride] = initial_confs[i][0].to(device)  # 224,224
        registered_confs_mean[i*kf_stride] = per_frame_res['l2w_confs'][i*kf_stride].mean().cpu()

    # initialize the buffering set with the initial window
    assert buffer_size <= 0 or buffer_size >= init_num 
    buffering_set_ids = [i*kf_stride for i in range(init_num)]
    
    # set up the world coordinates with frames in the initial window
    for i in range(init_num):
        input_views[i*kf_stride]['pts3d_world'] = initial_pcds[i]
        
    initial_valid_masks = [conf > conf_thres_i2p for conf in initial_confs] # 1,224,224
    normed_pts = normalize_views([view['pts3d_world'] for view in input_views[:init_num*kf_stride:kf_stride]],
                                                initial_valid_masks)
    for i in range(init_num):
        input_views[i*kf_stride]['pts3d_world'] = normed_pts[i]
        # filter out points with low confidence
        input_views[i*kf_stride]['pts3d_world'][~initial_valid_masks[i]] = 0       
        per_frame_res['l2w_pcds'][i*kf_stride] = normed_pts[i]  # 224,224,3

    # recover the pointmap of each view in their local coordinates with the I2P model
    # TODO: batchify
    local_confs_mean = []
    adj_distance = kf_stride
    for view_id in tqdm(range(num_views), desc="I2P resonstruction"):
        # skip the views in the initial window
        if view_id in buffering_set_ids:
            # trick to mark the keyframe in the initial window
            if view_id // kf_stride == init_ref_id:
                per_frame_res['i2p_pcds'][view_id] = per_frame_res['l2w_pcds'][view_id].cpu()
            else:
                per_frame_res['i2p_pcds'][view_id] = torch.zeros_like(per_frame_res['l2w_pcds'][view_id], device="cpu")
            per_frame_res['i2p_confs'][view_id] = per_frame_res['l2w_confs'][view_id].cpu()
            continue
        # construct the local window 
        sel_ids = [view_id]
        for i in range(1,win_r+1):
            if view_id-i*adj_distance >= 0:
                sel_ids.append(view_id-i*adj_distance)
            if view_id+i*adj_distance < num_views:
                sel_ids.append(view_id+i*adj_distance)
        local_views = [input_views[id] for id in sel_ids]
        ref_id = 0 
        # recover points in the local window, and save the keyframe points and confs
        output = i2p_inference_batch([local_views], i2p_model, ref_id=ref_id, 
                                    tocpu=False, unsqueeze=False)['preds']
        #save results of the i2p model
        per_frame_res['i2p_pcds'][view_id] = output[ref_id]['pts3d'].cpu() # 1,224,224,3
        per_frame_res['i2p_confs'][view_id] = output[ref_id]['conf'][0].cpu() # 224,224

        # construct the input for L2W model        
        input_views[view_id]['pts3d_cam'] = output[ref_id]['pts3d'] # 1,224,224,3
        valid_mask = output[ref_id]['conf'] > conf_thres_i2p # 1,224,224
        input_views[view_id]['pts3d_cam'] = normalize_views([input_views[view_id]['pts3d_cam']],
                                                    [valid_mask])[0]
        input_views[view_id]['pts3d_cam'][~valid_mask] = 0 

    local_confs_mean = [conf.mean() for conf in per_frame_res['i2p_confs']] # 224,224
    print(f'finish recovering pcds of {len(local_confs_mean)} frames in their local coordinates, with a mean confidence of {torch.stack(local_confs_mean).mean():.2f}')

    # Special treatment: register the frames within the range of initial window with L2W model
    # TODO: batchify
    if kf_stride > 1:
        max_conf_mean = -1
        for view_id in tqdm(range((init_num-1)*kf_stride), desc="pre-registering"):  
            if view_id % kf_stride == 0:
                continue
            # construct the input for L2W model
            l2w_input_views = [input_views[view_id]] + [input_views[id] for id in buffering_set_ids]
            # (for defination of ref_ids, see the doc of l2w_model)
            output = l2w_inference(l2w_input_views, l2w_model, 
                                   ref_ids=list(range(1,len(l2w_input_views))), 
                                   device=device,
                                   normalize=False)
            
            # process the output of L2W model
            input_views[view_id]['pts3d_world'] = output[0]['pts3d_in_other_view'] # 1,224,224,3
            conf_map = output[0]['conf'] # 1,224,224
            per_frame_res['l2w_confs'][view_id] = conf_map[0] # 224,224
            registered_confs_mean[view_id] = conf_map.mean().cpu()
            per_frame_res['l2w_pcds'][view_id] = input_views[view_id]['pts3d_world']
            
            if registered_confs_mean[view_id] > max_conf_mean:
                max_conf_mean = registered_confs_mean[view_id]
        print(f'finish aligning {(init_num-1)*kf_stride} head frames, with a max mean confidence of {max_conf_mean:.2f}')
        
        # A problem is that the registered_confs_mean of the initial window is generated by I2P model,
        # while the registered_confs_mean of the frames within the initial window is generated by L2W model,
        # so there exists a gap. Here we try to align it.
        max_initial_conf_mean = -1
        for i in range(init_num):
            if registered_confs_mean[i*kf_stride] > max_initial_conf_mean:
                max_initial_conf_mean = registered_confs_mean[i*kf_stride]
        factor = max_conf_mean/max_initial_conf_mean
        # print(f'align register confidence with a factor {factor}')
        for i in range(init_num):
            per_frame_res['l2w_confs'][i*kf_stride] *= factor
            registered_confs_mean[i*kf_stride] = per_frame_res['l2w_confs'][i*kf_stride].mean().cpu()

    # register the rest frames with L2W model
    next_register_id = (init_num-1)*kf_stride+1 # the next frame to be registered
    milestone = (init_num-1)*kf_stride+1 # All frames before milestone have undergone the selection process for entry into the buffering set.
    num_register = max(1,min((kf_stride+1)//2, 10))   # how many frames to register in each round
    # num_register = 1
    update_buffer_intv = kf_stride*update_buffer_intv   # update the buffering set every update_buffer_intv frames
    max_buffer_size = buffer_size
    strategy = buffer_strategy
    candi_frame_id = len(buffering_set_ids) # used for the reservoir sampling strategy
    
    pbar = tqdm(total=num_views, desc="registering")
    pbar.update(next_register_id-1)

    del i
    while next_register_id < num_views:
        ni = next_register_id
        max_id = min(ni+num_register, num_views)-1  # the last frame to be registered in this round

        # select sccene frames in the buffering set to work as a global reference
        cand_ref_ids = buffering_set_ids
        ref_views, sel_pool_ids = scene_frame_retrieve(
            [input_views[i] for i in cand_ref_ids], 
            input_views[ni:ni+num_register:2], 
            i2p_model, sel_num=num_scene_frame, 
            # cand_recon_confs=[per_frame_res['l2w_confs'][i] for i in cand_ref_ids],
            depth=2)
        
        # register the source frames in the local coordinates to the world coordinates with L2W model
        l2w_input_views = ref_views + input_views[ni:max_id+1]
        input_view_num = len(ref_views) + max_id - ni + 1
        assert input_view_num == len(l2w_input_views)
        
        output = l2w_inference(l2w_input_views, l2w_model, 
                               ref_ids=list(range(len(ref_views))), 
                               device=device,
                               normalize=False)
    
        # process the output of L2W model
        src_ids_local = [id+len(ref_views) for id in range(max_id-ni+1)]  # the ids of src views in the local window
        src_ids_global = [id for id in range(ni, max_id+1)]    #the ids of src views in the whole dataset
        succ_num = 0
        for id in range(len(src_ids_global)):
            output_id = src_ids_local[id] # the id of the output in the output list
            view_id = src_ids_global[id]    # the id of the view in all views
            conf_map = output[output_id]['conf'] # 1,224,224
            input_views[view_id]['pts3d_world'] = output[output_id]['pts3d_in_other_view'] # 1,224,224,3
            per_frame_res['l2w_confs'][view_id] = conf_map[0]
            registered_confs_mean[view_id] = conf_map[0].mean().cpu()
            per_frame_res['l2w_pcds'][view_id] = input_views[view_id]['pts3d_world']
            succ_num += 1
        # TODO:refine scene frames together
        # for j in range(1, input_view_num):
            # views[i-j]['pts3d_world'] = output[input_view_num-1-j]['pts3d'].permute(0,3,1,2)

        next_register_id += succ_num
        pbar.update(succ_num) 
        
        # update the buffering set
        if next_register_id - milestone >= update_buffer_intv:  
            while(next_register_id - milestone >= kf_stride):
                candi_frame_id += 1
                full_flag = max_buffer_size > 0 and len(buffering_set_ids) >= max_buffer_size
                insert_flag = (not full_flag) or ((strategy == 'fifo') or 
                                                  (strategy == 'reservoir' and np.random.rand() < max_buffer_size/candi_frame_id))
                if not insert_flag: 
                    milestone += kf_stride
                    continue
                # Use offest to ensure the selected view is not too close to the last selected view
                # If the last selected view is 0, 
                # the next selected view should be at least kf_stride*3//4 frames away
                start_ids_offset = max(0, buffering_set_ids[-1]+kf_stride*3//4 - milestone)
                    
                # get the mean confidence of the candidate views
                mean_cand_recon_confs = torch.stack([registered_confs_mean[i]
                                         for i in range(milestone+start_ids_offset, milestone+kf_stride)])
                mean_cand_local_confs = torch.stack([local_confs_mean[i]
                                         for i in range(milestone+start_ids_offset, milestone+kf_stride)])
                # normalize the confidence to [0,1], to avoid overconfidence
                mean_cand_recon_confs = (mean_cand_recon_confs - 1)/mean_cand_recon_confs # transform to sigmoid
                mean_cand_local_confs = (mean_cand_local_confs - 1)/mean_cand_local_confs
                # the final confidence is the product of the two kinds of confidences
                mean_cand_confs = mean_cand_recon_confs*mean_cand_local_confs
                
                most_conf_id = mean_cand_confs.argmax().item()
                most_conf_id += start_ids_offset
                id_to_buffer = milestone + most_conf_id
                buffering_set_ids.append(id_to_buffer)
                # print(f"add ref view {id_to_buffer}")                
                # since we have inserted a new frame, overflow must happen when full_flag is True
                if full_flag:
                    if strategy == 'reservoir':
                        buffering_set_ids.pop(np.random.randint(max_buffer_size))
                    elif strategy == 'fifo':
                        buffering_set_ids.pop(0)
                # print(next_register_id, buffering_set_ids)
                milestone += kf_stride
        # transfer the data to cpu if it is not in the buffering set, to save gpu memory
        for i in range(next_register_id):
            to_device(input_views[i], device=device if i in buffering_set_ids else 'cpu')
    
    pbar.close()
    
    fail_view = {}
    for i,conf in enumerate(registered_confs_mean):
        if conf < 10:
            fail_view[i] = conf.item()
    print(f'mean confidence for whole scene reconstruction: {torch.tensor(registered_confs_mean).mean().item():.2f}')
    print(f"{len(fail_view)} views with low confidence: ", {key:round(fail_view[key],2) for key in fail_view.keys()})

    per_frame_res['rgb_imgs'] = rgb_imgs

    save_path = get_model_from_scene(per_frame_res=per_frame_res,
                                     save_dir=save_dir,
                                     num_points_save=num_points_save,
                                     conf_thres_res=conf_thres_l2w,
                                     save_traj=save_traj)

    return save_path, per_frame_res
    
    
def get_model_from_scene(per_frame_res, save_dir,
                         num_points_save=200000,
                         conf_thres_res=3,
                         valid_masks=None,
                         save_traj=False
                        ):

    # collect the registered point clouds and rgb colors
    pcds = []
    rgbs = []
    pred_frame_num = len(per_frame_res['l2w_pcds'])
    registered_confs = per_frame_res['l2w_confs']
    registered_pcds = per_frame_res['l2w_pcds']
    rgb_imgs = per_frame_res['rgb_imgs']
    for i in range(pred_frame_num):
        registered_pcd = to_numpy(registered_pcds[i])
        if registered_pcd.shape[0] == 3:
            registered_pcd = registered_pcd.transpose(1,2,0)
        registered_pcd = registered_pcd.reshape(-1,3)
        rgb = rgb_imgs[i].reshape(-1,3)
        pcds.append(registered_pcd)
        rgbs.append(rgb)

    res_pcds = np.concatenate(pcds, axis=0)
    res_rgbs = np.concatenate(rgbs, axis=0)

    pts_count = len(res_pcds)
    valid_ids = np.arange(pts_count)

    # filter out points with gt valid masks
    if valid_masks is not None:
        valid_masks = np.stack(valid_masks, axis=0).reshape(-1)
        # print('filter out ratio of points by gt valid masks:', 1.-valid_masks.astype(float).mean())
    else:
        valid_masks = np.ones(pts_count, dtype=bool)

    # filter out points with low confidence
    if registered_confs is not None:
        conf_masks = []
        for i in range(len(registered_confs)):
            conf = registered_confs[i]
            conf_mask = (conf > conf_thres_res).reshape(-1).cpu()
            conf_masks.append(conf_mask)
        conf_masks = np.array(torch.cat(conf_masks))
        valid_ids = valid_ids[conf_masks&valid_masks]
        print('ratio of points filered out: {:.2f}%'.format((1.-len(valid_ids)/pts_count)*100))

    # sample from the resulting pcd consisting of all frames
    n_samples = min(num_points_save, len(valid_ids))
    print(f"resampling {n_samples} points from {len(valid_ids)} points")
    sampled_idx = np.random.choice(valid_ids, n_samples, replace=False)
    sampled_pts = res_pcds[sampled_idx]
    sampled_rgbs = res_rgbs[sampled_idx]
    sampled_pts[..., 1:] *= -1 # flip the axis for better visualization

    save_name = f"recon.glb"
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=sampled_pts, colors=sampled_rgbs/255.))
    save_path = join(save_dir, save_name)

    # save camera trajectory
    if save_traj:
        try:
            # estimate intrinsics for each frame
            intrinsics = []
            for i in range(pred_frame_num):
                pts3d_local = per_frame_res['i2p_pcds'][i]
                # Add batch dimension if needed (estimate_intrinsics expects 4D input)
                if len(pts3d_local.shape) == 3:
                    pts3d_local = pts3d_local.unsqueeze(0)
                intrinsic = estimate_intrinsics(pts3d_local)
                intrinsics.append(intrinsic)
            
            # save trajectory
            c2ws = save_traj_file(per_frame_res, pred_frame_num, save_dir, 
                          scene_id="scene", intrinsics=intrinsics)
            
            if c2ws is not None:
                # 根据场景大小算出合适的摄像机显示比例
                scale = np.percentile(np.linalg.norm(sampled_pts, axis=-1), 90) * 0.05
                scale = max(0.01, scale)
                for c2w in c2ws:
                    # 相机外壳金字塔
                    vertices = np.array([
                        [0, 0, 0],
                        [-scale*1.33, -scale, scale*2],
                        [scale*1.33, -scale, scale*2],
                        [scale*1.33, scale, scale*2],
                        [-scale*1.33, scale, scale*2],
                    ])
                    faces = np.array([
                        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 3, 2], [1, 4, 3]
                    ])
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    mesh.visual.vertex_colors = [255, 0, 0, 255] # 红色
                    mesh.apply_transform(c2w)
                    
                    # 匹配前段对 sampled_pts 所做的坐标系翻转
                    flip_mat = np.eye(4)
                    flip_mat[1, 1] = -1
                    flip_mat[2, 2] = -1
                    mesh.apply_transform(flip_mat)
                    
                    scene.add_geometry(mesh)
            print(f"Camera trajectory saved to {save_dir}")
        except Exception as e:
            print(f"Failed to save camera trajectory: {e}")

    scene.export(save_path)
    return save_path


def save_traj_file(views, pred_frame_num, save_dir, scene_id, intrinsics=None):
    """Save camera trajectory to a txt file.
    
    Args:
        views: list of view dictionaries containing pts3d_world
        pred_frame_num: number of frames to save
        save_dir: directory to save the trajectory file
        scene_id: prefix for the trajectory filename
        intrinsics: list of camera intrinsic matrices (3x3) for each frame
    """
    import cv2
    
    save_name = f"{scene_id}_traj.txt"
    
    c2ws = []
    H, W, _ = views['l2w_pcds'][0][0].shape
    
    for i in tqdm(range(pred_frame_num), desc="Saving camera trajectory"):
        pts = to_numpy(views['l2w_pcds'][i][0])
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        points_2d = np.stack((u, v), axis=-1)
        dist_coeffs = np.zeros(4).astype(np.float32)
        
        # Use PnP-RANSAC to estimate camera pose
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            pts.reshape(-1, 3).astype(np.float32),
            points_2d.reshape(-1, 2).astype(np.float32),
            intrinsics[i].astype(np.float32),
            dist_coeffs)
        
        if not success:
            print(f"Warning: Failed to estimate pose for frame {i}")
            c2ws.append(np.eye(4))
            continue
            
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        # Extrinsic parameters (4x4 matrix)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
        extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))
        c2w = inv(extrinsic_matrix)
        c2ws.append(c2w)
    
    c2ws = np.stack(c2ws, axis=0)
    translations = c2ws[:, :3, 3]
    
    # draw the trajectory in horizontal plane
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_traj(ax, [i for i in range(len(translations))], translations,
              '-', "black", "estimate trajectory")
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.savefig(join(save_dir, save_name.replace('.txt', '.png')), dpi=90)
    plt.close()
    
    # Save trajectory as txt file (each row is a 4x4 matrix flattened to 16 values)
    np.savetxt(join(save_dir, save_name), c2ws.reshape(-1, 16))
    print(f"Trajectory saved to {join(save_dir, save_name)}")
    
    return c2ws


def plot_traj(ax, stamps, traj, style, color, label):
    """Plot a trajectory using matplotlib."""
    stamps.sort()
    interval = np.median([s-t for s, t in zip(stamps[1:], stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x) > 0:
            ax.plot(x, y, style, color=color, label=label)
            label = ""
            x = []
            y = []
        last = stamps[i]
    if len(x) > 0:
        ax.plot(x, y, style, color=color, label=label)


def display_inputs(images):
    img_label = "Click or use the left/right arrow keys to browse images", 

    if images is None or len(images) == 0:
        return [gradio.update(label=img_label, value=None, visible=False, selected_index=0, scale=2, preview=True, height=300,),
                gradio.update(value=None, visible=False, scale=2, height=300,)]  

    if isinstance(images, str): 
        file_path = images
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        if any(file_path.endswith(ext) for ext in video_extensions):
            return [gradio.update(label=img_label, value=None, visible=False, selected_index=0, scale=2, preview=True, height=300,),
                gradio.update(value=file_path, autoplay=True, visible=True, scale=2, height=300,)]
        else:
            return [gradio.update(label=img_label, value=None, visible=False, selected_index=0, scale=2, preview=True, height=300,),
                    gradio.update(value=None, visible=False, scale=2, height=300,)] 
            
    return [gradio.update(label=img_label, value=images, visible=True, selected_index=0, scale=2, preview=True, height=300,),
            gradio.update(value=None, visible=False, scale=2, height=300,)]

def change_inputfile_type(input_type):
    if input_type == "directory":
        inputfiles = gradio.File(file_count="directory", file_types=["image"],
                                 scale=2,
                                 value=[],
                                 label="Select a directory containing images")
        video_extract_fps = gradio.Number(value=5,
                                          scale=0,
                                          interactive=True,
                                          visible=False,
                                          label="fps for extracting frames from video")
    elif input_type == "images":
        inputfiles = gradio.File(file_count="multiple", file_types=["image"],
                                 scale=2,
                                 value=[],
                                 label="Upload multiple images")
        video_extract_fps = gradio.Number(value=5,
                                          scale=0,
                                          interactive=True,
                                          visible=False,
                                          label="fps for extracting frames from video")
    elif input_type == "video":
        inputfiles = gradio.File(file_count="single", file_types=["video"],
                                 scale=2,
                                 value=None,
                                 label="Upload a mp4 video")
        video_extract_fps = gradio.Number(value=5,
                                          interactive=True,
                                          scale=1,
                                          visible=True,
                                          label="fps for extracting frames from video")
    return inputfiles, video_extract_fps
    
def change_kf_stride_type(kf_stride, inputfiles, win_r):
    max_kf_stride = 10
    if kf_stride == "auto":
        kf_stride_fix = gradio.Slider(value=-1,minimum=-1, maximum=-1, step=1, 
                                      visible=False, interactive=True, 
                                      label="stride between keyframes",
                                      info="For I2P reconstruction!")
    elif kf_stride == "manual setting":
        kf_stride_fix = gradio.Slider(value=1,minimum=1, maximum=max_kf_stride, step=1, 
                                      visible=True, interactive=True, 
                                      label="stride between keyframes",
                                      info="For I2P reconstruction!")
    return kf_stride_fix

def change_buffer_strategy(buffer_strategy):
    if buffer_strategy == "reservoir" or buffer_strategy == "fifo":
        buffer_size = gradio.Number(value=100, precision=0, minimum=1,
                                    interactive=True, 
                                    visible=True,
                                    label="size of the buffering set",
                                    info="For L2W reconstruction!")
    elif buffer_strategy == "unbounded":
        buffer_size = gradio.Number(value=10000, precision=0, minimum=1,
                                    interactive=True, 
                                    visible=False,
                                    label="size of the buffering set",
                                    info="For L2W reconstruction!")
    return buffer_size

def main_demo(i2p_model, l2w_model, device, tmpdirname, server_name, server_port):
    recon_scene_func = functools.partial(recon_scene, i2p_model, l2w_model, device)
    
    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="SLAM3R Demo") as demo:
        # scene state is save so that you can change num_points_save... without rerunning the inference
        per_frame_res = gradio.State(None)
        tmpdir_name = gradio.State(tmpdirname)
        
        gradio.HTML('<h2 style="text-align: center;">SLAM3R Demo</h2>')
        with gradio.Column():
            with gradio.Row():
                input_type = gradio.Dropdown([ "directory", "images", "video"],
                                             scale=1,
                                             value='directory', label="select type of input files")
                video_extract_fps = gradio.Number(value=5,
                                                  scale=0,
                                                  interactive=True,
                                                  visible=False,
                                                  label="fps for extracting frames from video")
                inputfiles = gradio.File(file_count="directory", file_types=["image"],
                                         scale=2,
                                         height=200,
                                         label="Select a directory containing images")
              
                image_gallery = gradio.Gallery(label="Click or use the left/right arrow keys to browse images",
                                            visible=False,
                                            selected_index=0,
                                            preview=True,   
                                            height=300,
                                            scale=2)
                video_gallery = gradio.Video(label="Uploaded Video",
                                            visible=False,
                                            height=300,
                                            scale=2)

            with gradio.Row():
                kf_stride = gradio.Dropdown(["auto", "manual setting"], label="how to choose stride between keyframes",
                                           value='auto', interactive=True,  
                                           info="For I2P reconstruction!")
                kf_stride_fix = gradio.Slider(value=-1, minimum=-1, maximum=-1, step=1, 
                                              visible=False, interactive=True, 
                                              label="stride between keyframes",
                                              info="For I2P reconstruction!")
                win_r = gradio.Number(value=5, precision=0, minimum=1, maximum=200,
                                      interactive=True, 
                                      label="the radius of the input window",
                                      info="For I2P reconstruction!")
                initial_winsize = gradio.Number(value=5, precision=0, minimum=2, maximum=200,
                                      interactive=True, 
                                      label="the number of frames for initialization",
                                      info="For I2P reconstruction!")
                conf_thres_i2p = gradio.Slider(value=1.5, minimum=1., maximum=10,
                                      interactive=True, 
                                      label="confidence threshold for the i2p model",
                                      info="For I2P reconstruction!")
            
            with gradio.Row():
                num_scene_frame = gradio.Slider(value=10, minimum=1., maximum=100, step=1,
                                      interactive=True, 
                                      label="the number of scene frames for reference",
                                      info="For L2W reconstruction!")
                buffer_strategy = gradio.Dropdown(["reservoir", "fifo","unbounded"], 
                                           value='reservoir', interactive=True,  
                                           label="strategy for buffer management",
                                           info="For L2W reconstruction!")
                buffer_size = gradio.Number(value=100, precision=0, minimum=1,
                                      interactive=True, 
                                      visible=True,
                                      label="size of the buffering set",
                                      info="For L2W reconstruction!")
                update_buffer_intv = gradio.Number(value=1, precision=0, minimum=1,
                                      interactive=True, 
                                      label="the interval of updating the buffering set",
                                      info="For L2W reconstruction!")
            
            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence threshold
                conf_thres_l2w = gradio.Slider(value=12, minimum=1., maximum=100,
                                      interactive=True,
                                      label="confidence threshold for the result",
                                      )
                # adjust the camera size in the output pointcloud
                num_points_save = gradio.Number(value=1000000, precision=0, minimum=1,
                                      interactive=True,
                                      label="number of points sampled from the result",
                                      )
                # option to save camera trajectory
                save_traj = gradio.Checkbox(value=False,
                                      interactive=True,
                                      label="Save camera trajectory (txt + png)",
                                      )

            outmodel = gradio.Model3D(height=500,
                                      clear_color=(0.,0.,0.,0.3))

            # events
            inputfiles.change(display_inputs,
                                inputs=[inputfiles],
                                outputs=[image_gallery, video_gallery])
            input_type.change(change_inputfile_type,
                                inputs=[input_type],
                                outputs=[inputfiles, video_extract_fps])
            kf_stride.change(change_kf_stride_type,
                                inputs=[kf_stride, inputfiles, win_r],
                                outputs=[kf_stride_fix])
            buffer_strategy.change(change_buffer_strategy,
                                inputs=[buffer_strategy],
                                outputs=[buffer_size])
            run_btn.click(fn=recon_scene_func,
                          inputs=[tmpdir_name, video_extract_fps,
                                  inputfiles, kf_stride_fix, win_r, initial_winsize, conf_thres_i2p,
                                  num_scene_frame, update_buffer_intv, buffer_strategy, buffer_size,
                                  conf_thres_l2w, num_points_save, save_traj],
                          outputs=[outmodel, per_frame_res])
            conf_thres_l2w.release(fn=get_model_from_scene,
                                 inputs=[per_frame_res, tmpdir_name, num_points_save, conf_thres_l2w, save_traj],
                                 outputs=outmodel)
            num_points_save.change(fn=get_model_from_scene,
                            inputs=[per_frame_res, tmpdir_name, num_points_save, conf_thres_l2w, save_traj],
                            outputs=outmodel)
            save_traj.change(fn=get_model_from_scene,
                            inputs=[per_frame_res, tmpdir_name, num_points_save, conf_thres_l2w, save_traj],
                            outputs=outmodel)

    demo.launch(share=False, server_name=server_name, server_port=server_port)


def main_offline(parser: argparse.ArgumentParser):
    
    args = parser.parse_args()

    if args.tmp_dir is not None:
        tmp_path = args.tmp_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'
    
    i2p_model = Image2PointsModel.from_pretrained('siyan824/slam3r_i2p')
    l2w_model = Local2WorldModel.from_pretrained('siyan824/slam3r_l2w')
    i2p_model.to(args.device)
    l2w_model.to(args.device)
    i2p_model.eval()
    l2w_model.eval()

    # slam3r will write the 3D model inside tmpdirname
    with tempfile.TemporaryDirectory(suffix='slam3r_gradio_demo') as tmpdirname:
        main_demo(i2p_model, l2w_model, args.device, tmpdirname, server_name, args.server_port)

if __name__ == "__main__":
    main_offline(argparse.ArgumentParser())