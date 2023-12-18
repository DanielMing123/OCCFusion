import pickle
import cv2
import os
import numpy as np
import torch
import multiprocessing
from vedo import *
from easydict import EasyDict
import torch.nn.functional as F
import shutil

def generate_pickle(pickle_path):
    
    res = dict()
    with open(pickle_path,'rb') as f:
        content = pickle.load(f)
        # import pdb;pdb.set_trace()
        res['metadata'] = content['metadata']
        res['infos'] = content['infos'][:1000]
        # res['metainfo'] = content['metainfo']
        # res['data_list'] = []
        # for info in content['data_list']:
        #     if 'n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530412470.jpg' == info['images']['CAM_FRONT']['img_path']:
        #         res['data_list'].append(info)
        #         break
    
    #     res['metainfo'] = content['metainfo']
    #     all_data = content['data_list']
    #     part_data = all_data[0:len(all_data):15]
    #     res['data_list'] = part_data
        
    with open('/home/daniel/BEV_3D_Occupancy/3D-OCCFusion/data/nuscenes/TEHI_nuscenes_infos_1000_con_train.pkl','wb') as out:
        pickle.dump(res,out)
         
def process_occ(folder_path, npy_file, save_folder):
    npy_path = os.path.join(folder_path,npy_file)
    occ = np.load(npy_path)
    occ[...,0] //= 2
    occ[...,1] //= 2
    occ[...,2] //= 2
    uniq_voxel, inv_ind, voxel_counts = np.unique(occ[:,:3], axis=0, return_inverse=True, return_counts=True)
    gt_label = np.zeros((uniq_voxel.shape[0],1),dtype=np.int32)
    for i in range(len(uniq_voxel)):
        labels = occ[inv_ind == i, 3]
        unique_labels, counts = np.unique(labels, return_counts=True)
        gt_label[i] = unique_labels[counts.argmax()]
    res_occ = np.concatenate([uniq_voxel, gt_label], axis=-1)
    save_path = os.path.join(save_folder, npy_file)
    np.save(save_path, res_occ)

def generate_occ(folder_path, save_folder):
    npy_files = os.listdir(folder_path)
    pool = multiprocessing.Pool(processes = 20)
    for npy_file in npy_files:
        pool.apply_async(process_occ,[folder_path, npy_file, save_folder])
    pool.close()
    pool.join()
        
def read_pickle(pikl_path):
    occ_sample_folder = '/home/daniel/Downloads/nuscenes_occ/samples'
    with open(pikl_path,'rb') as f:
        content = pickle.load(f)
        for sample in content['data_list']:
            occ_file = sample['lidar_points']['lidar_path'] + '.npy'
            occ_path = os.path.join(occ_sample_folder,occ_file)
            occ = np.load(occ_path)
            occ = occ.astype(np.int32)
            occ = torch.from_numpy(occ)
            occ_gt = 255 * torch.ones((200,200,16),dtype=torch.int32)
            occ_gt[occ[...,0],occ[...,1],occ[...,2]] = occ[...,3]
            import pdb;pdb.set_trace()

def visualize(pkl_path,pred_folder, save_folder):
    # img_names = sorted(os.listdir(pred_folder))
    
    with open(pkl_path,'rb') as f:
        content = pickle.load(f)
        for i, data_sample in enumerate(content['data_list']):
            # if i+1 > len(img_names):
            #     return None
            cam_front_path = data_sample['images']['CAM_FRONT']['img_path']
            cam_front_right_path = data_sample['images']['CAM_FRONT_RIGHT']['img_path']
            cam_front_left_path = data_sample['images']['CAM_FRONT_LEFT']['img_path']
            cam_back_path = data_sample['images']['CAM_BACK']['img_path']
            cam_back_left_path = data_sample['images']['CAM_BACK_LEFT']['img_path']
            cam_back_right_path = data_sample['images']['CAM_BACK_RIGHT']['img_path']
            
            # import pdb;pdb.set_trace()
            # grid_img = cv2.imread(os.path.join(pred_folder,f'{i}.png'))
            img_front = cv2.imread(os.path.join('data/nuscenes/samples/CAM_FRONT',cam_front_path))  
            img_front_right = cv2.imread(os.path.join('data/nuscenes/samples/CAM_FRONT_RIGHT',cam_front_right_path))  
            img_front_left = cv2.imread(os.path.join('data/nuscenes/samples/CAM_FRONT_LEFT',cam_front_left_path))
            img_back = cv2.imread(os.path.join('data/nuscenes/samples/CAM_BACK',cam_back_path))
            img_back_left = cv2.imread(os.path.join('data/nuscenes/samples/CAM_BACK_LEFT',cam_back_left_path))
            img_back_right = cv2.imread(os.path.join('data/nuscenes/samples/CAM_BACK_RIGHT',cam_back_right_path))
            front = cv2.hconcat([img_front_left,img_front,img_front_right])
            back = cv2.hconcat([img_back_left,img_back,img_back_right])
            img = cv2.vconcat([front, back])
            img = cv2.resize(img,(2560,1440))
            # res = cv2.vconcat([img,grid_img])
            # res = cv2.resize(res,None,fx=0.5,fy=0.5)
            cv2.imwrite(os.path.join(save_folder,f'{i}.png'),img) # res

def multiscale_supervision(gt_occ, ratio, gt_shape):
    '''
    change ground truth shape as (B, W, H, Z) for each level supervision
    '''
    # import pdb;pdb.set_trace()
    gt = torch.zeros([gt_shape[0], gt_shape[1], gt_shape[2]]).to(gt_occ.device).type(torch.long) 
    coords_x = gt_occ[:, 0].to(torch.float) // ratio[0]
    coords_y = gt_occ[:, 1].to(torch.float) // ratio[1]
    coords_z = gt_occ[:, 2].to(torch.float) // ratio[2]
    coords_x = coords_x.to(torch.long)
    coords_y = coords_y.to(torch.long)
    coords_z = coords_z.to(torch.long)
    coords = torch.stack([coords_x,coords_y,coords_z],dim=1)
    gt[coords[:, 0], coords[:, 1], coords[:, 2]] =  gt_occ[:, 3]
    
    return gt

def vis_Occ_GT(pickle_path):
    with open(pickle_path,'rb') as f:
        content = pickle.load(f)
        for i, data_sample in enumerate(content['data_list']):
            occ_file_name = data_sample['lidar_points']['lidar_path'].split('/')[-1] + '.npy'
            occ_folder = '/home/daniel/BEV_3D_Occupancy/3D-OCCFusion/data/nuscenes/occ_samples'
            occ_path = os.path.join(occ_folder, occ_file_name)
            occ = np.load(occ_path)
            occ = torch.from_numpy(occ)
            # import pdb;pdb.set_trace()
            # voxels_256 = multiscale_supervision(occ.clone(),[0.78125,0.5],np.array([256,256,32],dtype=np.int32)) 
            voxels_128 = multiscale_supervision(occ,[1.5625,1.5625,1],np.array([128,128,16],dtype=np.int32))
            voxels_64 = multiscale_supervision(occ,[3.125,3.125,2],np.array([64,64,8],dtype=np.int32))
            voxels_32 = multiscale_supervision(occ,[6.25,6.25,4],np.array([32,32,4],dtype=np.int32))
            
            voxels = 255 * torch.ones([200,200,16],dtype=torch.long) # [200,200,16]
            voxels[occ[:,0], occ[:,1], occ[:,2]] = occ[:,3]
            # # import pdb;pdb.set_trace()
            voxels = voxels.unsqueeze(0).unsqueeze(1).float() # .unsqueeze(0).unsqueeze(1)
            voxels_256 =F.interpolate(voxels,size=(256,256,32))
            
            voxels_256 = voxels_256.squeeze().round().long()
            
            draw_with_vedo(voxels_256, save_path='/media/daniel/dataset/visualize/save_gt/save1',voxel_size=[0.390625,0.390625,0.25]) # [0.390625,0.390625,0.25]
            draw_with_vedo(voxels_128, save_path='/media/daniel/dataset/visualize/save_gt/save2',voxel_size=[0.78125,0.78125,0.5]) # [0.78125,0.78125,0.5]
            draw_with_vedo(voxels_64, save_path='/media/daniel/dataset/visualize/save_gt/save3',voxel_size=[1.5625,1.5625,1]) # [1.5625,1.5625,1]
            draw_with_vedo(voxels_32, save_path='/media/daniel/dataset/visualize/save_gt/save4',voxel_size=[3.125,3.125,2]) # [3.125,3.125,2]

def draw_with_vedo(voxels, save_path=None, voxel_size = [0.390625, 0.390625, 0.25], offscreen=True):
    w, h, z = voxels.shape
    grid_coords = get_grid_coords([voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size) + np.array([-40,-40,-1], dtype=np.float32).reshape([1, 3]) # [-50,-50,-5]
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    
    # car_vox_range = np.array([
    #     [w//2 - 3, w//2 + 3],
    #     [h//2 - 3, h//2 + 3],
    #     [z//2 - 2 - 3, z//2 - 2 + 3]
    # ], dtype=np.int32)
    # car_x = np.arange(car_vox_range[0, 0], car_vox_range[0, 1])
    # car_y = np.arange(car_vox_range[1, 0], car_vox_range[1, 1])
    # car_z = np.arange(car_vox_range[2, 0], car_vox_range[2, 1])
    # car_xx, car_yy, car_zz = np.meshgrid(car_x, car_y, car_z)
    # car_label = 17*np.ones([6, 6, 6], dtype=np.int32)
    # car_grid = np.array([car_xx.flatten(), car_yy.flatten(), car_zz.flatten()]).T
    # car_indexes = car_grid[:, 0] * h * z + car_grid[:, 1] * z + car_grid[:, 2]
    # grid_coords[car_indexes, 3] = car_label.flatten()

    grid_coords[grid_coords[:, 3] == 255, 3] = 20 # empty is not gonna being draw
    grid_coords[grid_coords[:, 3] == 0, 3] = 20   # noise is not gonna being draw
    
    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
    ]
    # permute (y, x, z) to (x, y, z)
    fov_voxels[:, [0, 1]] = fov_voxels[:, [1, 0]]

    # build plotter
    plotter = Plotter(offscreen=offscreen)

    colors = get_color_palette()[:, :3]
    cube_size = sum(voxel_size) / 3 * 0.95
    
    pts_center = Points(fov_voxels[:, :3])
    cube = Cube(side=cube_size)
    # TODO: check the -1 index
    colors = colors[fov_voxels[:, 3].astype(np.int32) - 1]
    # put cube at each point, and adjust the surface settings to make it look nicer
    occ_scene = Glyph(pts_center, cube, c=colors).lighting(ambient=0.8, diffuse=0.2)

    # set camera position
    camera = EasyDict()
    camera.position = [  0.75131739, -35.08337438,  16.71378558]
    camera.focal_point = [  0.75131739, -34.21734897,  16.21378558]
    # camera.focal_point = [  0,0,0]
    camera.view_angle = 90.0 # 40 # 90
    camera.viewup = [0.0, 0.0, 1.0]
    camera.clipping_range = [0.01, 300.]

    # show and save
    plotter.show(occ_scene,camera=camera, size=(2560, 1440))
    if offscreen:
        save_folder = save_path # '/home/daniel/BEV_3D_Occupancy/3D-OCCFusion/vis_train_data'
        num = len(os.listdir(save_folder))
        save_path = os.path.join(save_folder,f'{num}.png')
        plotter.screenshot(save_path)
    plotter.clear()

def get_grid_coords(dims, resolution):
        
        g_xx = np.arange(0, dims[0])  # [0, 1, ..., 256]
        # g_xx = g_xx[::-1]
        g_yy = np.arange(0, dims[1])  # [0, 1, ..., 256]
        # g_yy = g_yy[::-1]
        g_zz = np.arange(0, dims[2])  # [0, 1, ..., 32]

        # Obtaining the grid with coords...
        xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
        coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T # .T
        coords_grid = coords_grid.astype(np.float32)
        resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

        coords_grid = (coords_grid * resolution) + resolution / 2

        return coords_grid

def get_color_palette():
        return np.array(
            [
                [255, 120, 50, 255],  # barrier              orange
                [255, 192, 203, 255],  # bicycle              pink
                [255, 255, 0, 255],  # bus                  yellow
                [0, 150, 245, 255],  # car                  blue
                [0, 255, 255, 255],  # construction_vehicle cyan
                [255, 127, 0, 255],  # motorcycle           dark orange
                [255, 0, 0, 255],  # pedestrian           red
                [255, 240, 150, 255],  # traffic_cone         light yellow
                [135, 60, 0, 255],  # trailer              brown
                [160, 32, 240, 255],  # truck                purple
                [255, 0, 255, 255],  # driveable_surface    dark pink
                # [175,   0,  75, 255],       # other_flat           dark red
                [139, 137, 137, 255],
                [75, 0, 75, 255],  # sidewalk             dard purple
                [150, 240, 80, 255],  # terrain              light green
                [230, 230, 250, 255],  # manmade              white
                [0, 175, 0, 255],  # vegetation           green
                [0, 0, 0, 0],  # ego car              dark cyan
                [255, 99, 71, 255],  # ego car
                [0, 191, 255, 255]  # ego car
            ]
        ).astype(np.uint8)

def generate_vis_res(npy_folder, save_path):
    npy_files = os.listdir(npy_folder)
    for i in range(len(npy_files)):
        npy_path = os.path.join(npy_folder,f'{i}.npy')
        pred_voxels = np.load(npy_path)
        draw_with_vedo(pred_voxels,save_path)

def merge_train_val(train_pkl_path,val_pkl_path):
    res = dict()
    content_train = None
    content_val = None
    with open(train_pkl_path,'rb') as f:
        content_train = pickle.load(f)
    with open(val_pkl_path,'rb') as f:
        content_val = pickle.load(f)
    # import pdb;pdb.set_trace()
    res['metainfo'] = content_train['metainfo']
    res['data_list'] = content_train['data_list'] + content_val['data_list']
    
    with open('/home/daniel/BEV_3D_Occupancy/3D-OCCFusion/data/nuscenes/nuscenes_infos_trainval.pkl','wb') as out:
        pickle.dump(res,out)

def merge_vis_res(img_folder,occ_folder,gt_folder,save_folder):
    imgs = os.listdir(img_folder)
    occs = os.listdir(occ_folder)
    gts = os.listdir(gt_folder)
    for idx in range(len(imgs)):
        # import pdb;pdb.set_trace()
        img_path = os.path.join(img_folder,f'{idx}.png')
        occ_path = os.path.join(occ_folder,f'{idx}.png')
        gt_path = os.path.join(gt_folder,f'{idx}.png')
        
        img = cv2.imread(img_path)
        occ = cv2.imread(occ_path)
        gt = cv2.imread(gt_path)
        res = cv2.vconcat([img, occ, gt])
        cv2.imwrite(os.path.join(save_folder,f'{idx}.png'),res)

def move_folders(ori_folder, tar_folder):
    contents = os.listdir(ori_folder)
    for content in contents:
        content_path = os.path.join(ori_folder,content)
        sub_folders = os.listdir(content_path)
        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(content_path,sub_folder)
            shutil.move(sub_folder_path,tar_folder)

def occ3d_visualize(pkl_path,occ_save_path):
    with open(pkl_path,'rb') as f:
        content = pickle.load(f)
        for i, data_sample in enumerate(content['data_list']):
            token = data_sample['token']
            occ_path = os.path.join('/media/daniel/Expansion/nuscenes/Occ3D', token, 'labels.npz')
            occ_3d = np.load(occ_path)
            occ_3d_semantic = occ_3d['semantics']
            occ_3d_cam_mask = occ_3d['mask_camera']
            occ_3d_gt = occ_3d_semantic * occ_3d_cam_mask
            occ_3d_gt[occ_3d_gt==0]=255
            occ_3d_gt[occ_3d_gt==17]=0
            occ_3d_gt = torch.from_numpy(occ_3d_gt)
            idx = torch.where(occ_3d_gt > 0)
            label = occ_3d_gt[idx[0],idx[1],idx[2]]
            occ_3d = torch.stack([idx[0],idx[1],idx[2],label],dim=1).float()
            rot_mat = torch.tensor([[np.cos(-np.pi/2), -np.sin(-np.pi/2)],
                                    [np.sin(-np.pi/2), np.cos(-np.pi/2)]]).float()
            occ_3d[:,0:2] = torch.mm(occ_3d[:,0:2], rot_mat)
            occ_3d = occ_3d.long()
            voxels_256 = multiscale_supervision(occ_3d,[1,1,1],np.array([200,200,16],dtype=np.int32))
            voxels_128 = multiscale_supervision(occ_3d,[2,2,2],np.array([100,100,8],dtype=np.int32))
            voxels_64 = multiscale_supervision(occ_3d,[4,4,4],np.array([50,50,4],dtype=np.int32))
            voxels_32 = multiscale_supervision(occ_3d,[8,8,8],np.array([25,25,2],dtype=np.int32))

            draw_with_vedo(voxels_256, save_path='/media/daniel/dataset/visualize/save_occ3d_gt/save1_mask',voxel_size=[0.4,0.4,0.4]) # [0.390625,0.390625,0.25]
            draw_with_vedo(voxels_128, save_path='/media/daniel/dataset/visualize/save_occ3d_gt/save2_mask',voxel_size=[0.8,0.8,0.8]) # [0.78125,0.78125,0.5]
            draw_with_vedo(voxels_64, save_path='/media/daniel/dataset/visualize/save_occ3d_gt/save3_mask',voxel_size=[1.6,1.6,1.6]) # [1.5625,1.5625,1]
            draw_with_vedo(voxels_32, save_path='/media/daniel/dataset/visualize/save_occ3d_gt/save4_mask',voxel_size=[3.2,3.2,3.2]) # [3.125,3.125,2]

if __name__ == '__main__':
    train_THEI_pkl_path = '/home/daniel/BEV_3D_Occupancy/The-Eyes-Have-It/data/TEHI_nuscenes_infos_train.pkl'
    val_THEI_pkl_path = '/home/daniel/BEV_3D_Occupancy/The-Eyes-Have-It/data/TEHI_nuscenes_infos_val.pkl'
    train_pkl_path = '/media/daniel/Expansion/nuscenes/nuscenes_infos_train.pkl'
    train_400_pkl_path = '/media/daniel/Expansion/nuscenes/nuscenes_infos_400_train.pkl'
    train_2000_pkl_path = '/media/daniel/Expansion/nuscenes/nuscenes_infos_2000_train.pkl'
    val_pkl_path = '/media/daniel/Expansion/nuscenes/nuscenes_infos_val.pkl'
    val_100_pkl_path = '/media/daniel/Expansion/nuscenes/nuscenes_infos_100_con_train.pkl'
    pred_folder = '/home/daniel/BEV_3D_Occupancy/3D-OCCFusion/predict'
    pkl_surrocc_path = '/home/daniel/Downloads/nuscenes_infos_train.pkl'
    occ_folder = '/media/daniel/Expansion/nuscenes/occ_samples'
    occ_save_folder = '/home/daniel/BEV_3D_Occupancy/3D-OCCFusion/data/occ_resize_sample'
    
    npy_folder = '/media/daniel/dataset/visualize/pred_occ_npy'
    occ_save_path = '/media/daniel/dataset/visualize/occ_res2'
    save_folder = '/home/daniel/BEV_3D_Occupancy/3D-OCCFusion/visualize/input_imgs'
    img_folder = '/media/daniel/dataset/visualize/input_imgs'
    occ_folder = '/media/daniel/dataset/visualize/occ_res'
    gt_folder = '/media/daniel/dataset/visualize/save_gt/save1'
    save_folder = '/media/daniel/dataset/visualize/merged_res_2'
    ori_folder = '/home/daniel/Downloads/Occ3D/gts'
    tar_folder = '/home/daniel/Downloads/Occ3D/all_label'
    occ3d_vis_result = '/media/daniel/dataset/visualize/save_occ3d_gt'
    
    occ3d_visualize(val_pkl_path, occ3d_vis_result)
    # move_folders(ori_folder,tar_folder)
    # generate_vis_res(npy_folder,occ_save_path)
    # visualize(val_pkl_path, occ_save_path, save_folder) 
    # merge_vis_res(img_folder,occ_folder,gt_folder,save_folder)
    # read_pickle(pkl_path)
    # generate_occ(occ_folder, occ_save_folder)
    # generate_pickle(val_THEI_pkl_path)
    # vis_save_path = '/home/daniel/BEV_3D_Occupancy/3D-OCCFusion/vis_400_ori_train_data'
    # vis_Occ_GT(val_pkl_path)
    # merge_train_val(train_pkl_path,val_pkl_path)