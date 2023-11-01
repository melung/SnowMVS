import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import norm
import numpy as np
from models.mvs4net_utils import stagenet, reg2d, reg2d_no_n, FPN4, FPN4_convnext, FPN4_convnext4, PosEncSine, PosEncLearned, NormalNet, ConfidenceNet, \
        init_range, schedule_range, init_inverse_range, schedule_inverse_range, sinkhorn, mono_depth_decoder, ASFF
class SurfaceNet(nn.Module):

    def __init__(self):
        super(SurfaceNet, self).__init__()



    def forward(self, x, ax, ay):
        self.device = x.device
        #start = timeit.default_timer()
        #x = x.to(self.device)
        with torch.no_grad():
            nb_channels = 1#x.shape[1]
            b, _, h, w = x.shape
            #print("input", x.size())

            delzdelxkernel = torch.FloatTensor([[0.00000, 0.00000, 0.00000],
                                            [-1.00000, 0.00000, 1.00000],
                                            [0.00000, 0.00000, 0.00000]])
            delzdelxkernel = delzdelxkernel.view(1, 1, 3, 3).repeat(nb_channels,nb_channels, 1, 1).to(self.device)


            delta_x = x/(ax) + 1e-6
            #print("delta_x",delta_x.size())
            delta_y = x/(ay) + 1e-6

            delzdelx = F.conv2d(x, delzdelxkernel,padding =1)/(2.0)
            delzdelx = delzdelx/delta_x

            delzdelykernel = torch.FloatTensor([[0.00000, -1.00000, 0.00000],
                                            [0.00000, 0.00000, 0.00000],
                                            [0.0000, 1.00000, 0.00000]])
            delzdelykernel = delzdelykernel.view(1, 1, 3, 3).repeat(nb_channels,nb_channels, 1, 1).to(self.device)
            #print("kerneal size",delzdelykernel.size())
            
            delzdely = F.conv2d(x, delzdelykernel,padding =1)/(2.0)
            delzdely = delzdely/delta_y


            delzdelz = torch.ones(delzdely.shape, dtype=torch.float64).to(self.device)
            #zz = torch.ones(yy.shape, dtype=torch.float64).to(self.device)

            #print('kernel',delzdelx.shape)
            surface_norm = torch.stack((-delzdelx,-delzdely, delzdelz),2)
            
            
            #curvature_norm = torch.stack((-xx,-yy, zz),2)

            #print("normal result1: ",surface_norm.size())
            
            surface_norm = torch.div(surface_norm,  norm(surface_norm, dim=2)[:,:,None,:,:]).float()
            
            #print("x",x.size())
            #print("surface norm size",surface_norm.size())
            
            cur_x = F.conv2d(surface_norm[:,:,0,:,:], delzdelxkernel,padding =1)/2.0
            cur_y = F.conv2d(surface_norm[:,:,1,:,:], delzdelykernel,padding =1)/2.0
            #print("curx size:",cur_x.size())
            mean_cur = torch.sum(torch.stack((cur_x,cur_y),1),dim=1)/2.0

            
            curvature = torch.abs(F.avg_pool2d(mean_cur.squeeze(1), kernel_size= 3, stride = 1, padding = 1))

            return surface_norm.squeeze(1) , curvature

class MVS4net(nn.Module):
    def __init__(self, arch_mode="fpn", reg_net='reg2d', num_stage=4, fpn_base_channel=8, 
                reg_channel=8, stage_splits=[8,8,4,4], depth_interals_ratio=[0.5,0.5,0.5,1],
                group_cor=False, group_cor_dim=[8,8,8,8],
                inverse_depth=False,
                agg_type='ConvBnReLU3D',
                dcn=False,
                pos_enc=0,
                mono=False,
                asff=False,
                attn_temp=2,
                attn_fuse_d=True,
                vis_ETA=False,
                vis_mono=False
                ):
        # pos_enc: 0 no pos enc; 1 depth sine; 2 learnable pos enc
        super(MVS4net, self).__init__()
        self.arch_mode = arch_mode
        self.num_stage = num_stage
        self.depth_interals_ratio = depth_interals_ratio
        self.group_cor = group_cor
        self.group_cor_dim = group_cor_dim
        self.inverse_depth = inverse_depth
        self.asff = asff
        if self.asff:
            self.asff = nn.ModuleList([ASFF(i) for i in range(num_stage)])
        self.attn_ob = nn.ModuleList()
        if arch_mode == "fpn":
            self.feature = FPN4(base_channels=fpn_base_channel, gn=False, dcn=dcn)
        #print("dcn :", dcn)
            
        self.vis_mono = vis_mono
        self.stagenet = stagenet(inverse_depth, mono, attn_fuse_d, vis_ETA, attn_temp)
        
        self.stage_splits = stage_splits
        if reg_net == 'reg2d':
            self.normalnet = nn.ModuleList([NormalNet(in_channels=self.stage_splits[i]) for i in range(self.num_stage)])
        elif reg_net == 'reg2d_no_n':
            self.normalnet = nn.ModuleList([SurfaceNet() for i in range(self.num_stage)])


        

        self.confinet = ConfidenceNet(input_depths=self.stage_splits[3], input_channel = self.group_cor_dim[3])
        self.curriculum_learning_rho_ratios = [9, 4, 2, 1]

        self.reg = nn.ModuleList()
        self.pos_enc = pos_enc
        self.pos_enc_func = nn.ModuleList()
        self.mono = mono
        if self.mono:
            self.mono_depth_decoder = mono_depth_decoder()
        if reg_net == 'reg3d':
            self.down_size = [3,3,2,2]
        for idx in range(num_stage):
            if self.group_cor:
                in_dim = group_cor_dim[idx]
            else:
                in_dim = self.feature.out_channels[idx]
            if reg_net == 'reg2d':
                self.reg.append(reg2d(input_channel=in_dim, base_channel=reg_channel, conv_name=agg_type))
            elif reg_net == 'reg2d_no_n':
                self.reg.append(reg2d_no_n(input_channel=in_dim, base_channel=reg_channel, conv_name=agg_type))


    def forward(self, imgs, rotation, proj_matrices, depth_values, filename=None):
        depth_min = depth_values[:, 0].cpu().numpy()
        depth_max = depth_values[:, -1].cpu().numpy()
        depth_interval = (depth_max - depth_min) / depth_values.size(1)
        
        # step 1. feature extraction
        features = []
        for nview_idx in range(len(imgs)):  #imgs shape (B, N, C, H, W)
            img = imgs[nview_idx]
            features.append(self.feature(img))
        if self.vis_mono:
            scan_name = filename[0].split('/')[0]
            image_name = filename[0].split('/')[2][:-2]
            save_fn = './debug_figs/vis_mono/feat_{}'.format(scan_name+'_'+image_name)
            feat_ = features[-1]['stage4'].detach().cpu().numpy()
            np.save(save_fn, feat_)
        # step 2. iter (multi-scale)
        outputs = {}
        for stage_idx in range(self.num_stage):
            if not self.asff:
                features_stage = [feat["stage{}".format(stage_idx+1)] for feat in features]
            else:
                features_stage = [self.asff[stage_idx](feat['stage1'],feat['stage2'],feat['stage3'],feat['stage4']) for feat in features]

            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            B,C,H,W = features[0]['stage{}'.format(stage_idx+1)].shape

            #print("Hw",H,W)
            # init range
            if stage_idx == 0:
                if self.inverse_depth:
                    depth_hypo, normal_plane = init_inverse_range(depth_values, self.stage_splits[stage_idx], img[0].device, img[0].dtype, H, W)
                else:
                    depth_hypo, normal_plane = init_range(depth_values, self.stage_splits[stage_idx], img[0].device, img[0].dtype, H, W)
                sum_weight = []
            else:
                if self.inverse_depth:
                    depth_hypo, normal_plane = schedule_inverse_range(outputs_stage['inverse_min_depth'].detach(), outputs_stage['inverse_max_depth'].detach(), outputs_stage['normal_plane'].detach(), self.stage_splits[stage_idx], H, W)  # B D H W
                else:
                    depth_hypo, normal_plane = schedule_range(outputs_stage['inverse_min_depth'].detach(), outputs_stage['inverse_max_depth'].detach(), outputs_stage['normal_plane'].detach(), self.stage_splits[stage_idx], H, W, sum_weight)

                
            outputs_stage = self.stagenet(features_stage, rotation, normal_plane, self.normalnet[stage_idx],self.confinet, proj_matrices_stage, depth_hypo=depth_hypo, regnet=self.reg[stage_idx], stage_idx=stage_idx,
                                        group_cor=self.group_cor, group_cor_dim=self.group_cor_dim[stage_idx],
                                        split_itv=self.depth_interals_ratio[stage_idx],
                                        fn=filename)
            sum_weight = outputs_stage['sum_weight']
            depth_est = outputs_stage['depth']
            #depth_est_filtered = frequency_domain_filter(depth_est, rho_ratio=self.curriculum_learning_rho_ratios[stage_idx])
            #outputs_stage['depth_filtered'] = depth_est_filtered
            outputs_stage['depth_filtered'] = depth_est
            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)
        
        #if self.mono and self.training:
        # if self.mono:
            #outputs = self.mono_depth_decoder(outputs, depth_values[:,0], depth_values[:,1])

        return outputs


def frequency_domain_filter(depth, rho_ratio):
    """
    large rho_ratio -> more information filtered
    """
    f = torch.fft.fft2(depth)
    fshift = torch.fft.fftshift(f)

    b, h, w = depth.shape
    k_h, k_w = h/rho_ratio, w/rho_ratio

    fshift[:,:int(h/2-k_h/2),:] = 0
    fshift[:,int(h/2+k_h/2):,:] = 0
    fshift[:,:,:int(w/2-k_w/2)] = 0
    fshift[:,:,int(w/2+k_w/2):] = 0

    ishift = torch.fft.ifftshift(fshift)
    idepth = torch.fft.ifft2(ishift)
    depth_filtered = torch.abs(idepth)

    return depth_filtered

# def MVS4net_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
#     surfacenet = SurfaceNet()
#     stage_lw = kwargs.get("stage_lw", [8,4,2,1])
#     l1ot_lw = kwargs.get("l1ot_lw", [0,1])
#     inverse = kwargs.get("inverse_depth", False)
#     ot_iter = kwargs.get("ot_iter", 3)
#     ot_eps = kwargs.get("ot_eps", 1)
#     ot_continous = kwargs.get("ot_continous", False)
#     mono = kwargs.get("mono", False)
#     total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
#     stage_ot_loss = []
#     stage_l1_loss = []
#     range_err_ratio = []
#     stage_normal_loss = []
#     for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
#         depth_pred = stage_inputs['depth']
#         hypo_depth = stage_inputs['hypo_depth']
#         attn_weight = stage_inputs['attn_weight']
#         normal_est = stage_inputs["normal_plane"]
#         confidence = stage_inputs['photometric_confidence']
#         #normal_est = normal_est[:,0:2,:,:]
#         ax = stage_inputs["ax"]
#         ay = stage_inputs["ay"]
        
        
#         B,H,W = depth_pred.shape
#         D = hypo_depth.shape[1]
#         mask = mask_ms[stage_key]
#         mask = mask > 0.5
#         normal_mask = mask.unsqueeze(1).repeat(1,3,1,1)
#         depth_gt = depth_gt_ms[stage_key]
#         normal_gt, _ = surfacenet(depth_gt.unsqueeze(1), ax, ay)
#         #normal_gt = normal_gt[:,0:2,:,:]
#         #if mono and stage_idx!=0:
#             #this_stage_l1_loss = F.l1_loss(stage_inputs['mono_depth'][mask], depth_gt[mask], reduction='mean')
#         #else:
        
#         #this_stage_l1_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

#         # mask range
#         if inverse:
#             depth_itv = (1/hypo_depth[:,2,:,:]-1/hypo_depth[:,1,:,:]).abs()  # B H W
#             mask_out_of_range = ((1/hypo_depth - 1/depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0 # B H W
#             #print(mask_out_of_range)
#             #print(mask_out_of_range.size())
#             # if stage_idx == 3:
#             #     this_stage_confidence_loss = 40*F.smooth_l1_loss(confidence[mask], mask_out_of_range[mask], reduction='mean')
#             # else:
#             this_stage_confidence_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
#         else:
#             depth_itv = (hypo_depth[:,-1,:,:]-hypo_depth[:,-2,:,:]).abs()  # B H W
#             mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0 # B H W
#             # if stage_idx == 3:
#             #     this_stage_confidence_loss = 40*F.smooth_l1_loss(confidence[mask], mask_out_of_range[mask], reduction='mean')
#             # else:
#             this_stage_confidence_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
        
#         #마지막 regression but mask 필요    
#         range_err_ratio.append(mask_out_of_range[mask].float().mean())
#         this_stage_normal_loss = 5*F.smooth_l1_loss(normal_est[normal_mask], normal_gt[normal_mask], reduction='mean')
#         # if stage_idx == 3:
#         #     this_stage_ot_loss = F.smooth_l1_loss(depth_pred[mask], depth_gt[mask], reduction='mean')
#         # else:
#         #     this_stage_ot_loss = sinkhorn(depth_gt, hypo_depth, attn_weight, mask, iters=ot_iter, eps=ot_eps, continuous=ot_continous)[1]
#         this_stage_ot_loss = sinkhorn(depth_gt, hypo_depth, attn_weight, mask, iters=ot_iter, eps=ot_eps, continuous=ot_continous)[1]

#         stage_l1_loss.append(this_stage_ot_loss)
#         stage_ot_loss.append(this_stage_confidence_loss)
#         stage_normal_loss.append(this_stage_normal_loss)
#         total_loss = total_loss + stage_lw[stage_idx] * (l1ot_lw[1] * this_stage_ot_loss ) + this_stage_normal_loss

#     return total_loss, stage_l1_loss, stage_ot_loss, range_err_ratio, stage_normal_loss


def MVS4net_loss(depth_values, inputs, depth_gt_ms, mask_ms, **kwargs):
    surfacenet = SurfaceNet()
    stage_lw = kwargs.get("stage_lw", [1,1,1,1])
    l1ot_lw = kwargs.get("l1ot_lw", [0,1])
    inverse = kwargs.get("inverse_depth", False)
    ot_iter = kwargs.get("ot_iter", 3)
    ot_eps = kwargs.get("ot_eps", 1)
    ot_continous = kwargs.get("ot_continous", False)
    mono = kwargs.get("mono", False)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    stage_ot_loss = []
    stage_l1_loss = []
    range_err_ratio = []
    stage_normal_loss = []
    depth_min, depth_max = depth_values[:,0], depth_values[:,-1]
    
    
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        #depth_pred = stage_inputs['depth']
        depth_pred = stage_inputs['depth_filtered']
        hypo_depth = stage_inputs['hypo_depth']
        prob_volume = stage_inputs['attn_weight']
        normal_est = stage_inputs["normal_plane"]
        
        #normal_est = normal_est[:,0:2,:,:]
        ax = stage_inputs["ax"]
        ay = stage_inputs["ay"]
        
        
        B,H,W = depth_pred.shape
        D = hypo_depth.shape[1]
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        normal_mask = mask.unsqueeze(1).repeat(1,3,1,1)
        depth_gt = depth_gt_ms[stage_key]
        normal_gt, _ = surfacenet(depth_gt.unsqueeze(1), ax, ay)


        pw_loss = pixel_wise_loss(prob_volume, depth_gt, mask, hypo_depth, normal_gt)
        #dds_loss = depth_distribution_similarity_loss(depth_pred, depth_gt, mask, depth_min, depth_max)
        depth_itv = (hypo_depth[:,-1,:,:]-hypo_depth[:,-2,:,:]).abs()  # B H W
        mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0 # B H W
        
        #print(mask_out_of_range[mask].size())

        #마지막 regression but mask 필요    
        range_err_ratio.append(mask_out_of_range[mask].float().mean())
        
        #마지막 regression but mask 필요    
        this_stage_normal_loss = F.smooth_l1_loss(normal_est[normal_mask], normal_gt[normal_mask], reduction='mean')    

        #print(pw_loss)    
        stage_l1_loss.append(pw_loss)
        stage_ot_loss.append(pw_loss)
        #tage_normal_loss.append(pw_loss)
        stage_normal_loss.append(this_stage_normal_loss)

        #total_loss = total_loss + stage_lw[stage_idx] * (( pw_loss))
        total_loss = total_loss + stage_lw[stage_idx] * (( pw_loss) + this_stage_normal_loss)

    return total_loss, stage_l1_loss, stage_ot_loss, range_err_ratio, stage_normal_loss









def pixel_wise_loss(prob_volume, depth_gt, mask, depth_value, normal_gt):
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1,2])+1e-12

    shape = depth_gt.shape

    depth_num = depth_value.shape[1]
    depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)

    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume+1e-12), dim=1).squeeze(1)
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image)
    nweight = ((1 - torch.exp(-5*normal_gt[:,2,:,:]))/(1 + torch.exp(-5*normal_gt[:,2,:,:]))) + 0.1
    
    # print("normal",nweight.size())
    # print(nweight.min())
    # print(nweight.max())

    #print("loss",masked_cross_entropy_image.size())
    w_masked_cross_entropy_image = nweight*masked_cross_entropy_image
    #w_masked_cross_entropy_image = masked_cross_entropy_image
    masked_cross_entropy = torch.sum(w_masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)
    
    pw_loss = masked_cross_entropy
    return pw_loss

def depth_distribution_similarity_loss(depth, depth_gt, mask, depth_min, depth_max):
    # print(depth.size())
    # print(depth_max)
    # print(depth_min)
    depth_norm = depth * 128 / (depth_max - depth_min)[:,None,None]
    depth_gt_norm = depth_gt * 128 / (depth_max - depth_min)[:,None,None]
    M_bins = 48
    kl_min = torch.min(torch.min(depth_gt), depth.mean()-3.*depth.std())
    kl_max = torch.max(torch.max(depth_gt), depth.mean()+3.*depth.std())
    bins = torch.linspace(kl_min, kl_max, steps=M_bins)

    kl_divs = []
    for i in range(len(bins) - 1):
        bin_mask = (depth_gt >= bins[i]) & (depth_gt < bins[i+1])
        merged_mask = mask & bin_mask 

        if merged_mask.sum() > 0:
            p = depth_norm[merged_mask]
            q = depth_gt_norm[merged_mask]
            kl_div = torch.nn.functional.kl_div(torch.log(p)-torch.log(q), p, reduction='batchmean')
            kl_div = torch.log(kl_div)
            kl_divs.append(kl_div)

    dds_loss = sum(kl_divs)
    return dds_loss



def Blend_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1,1,1,1])
    l1ot_lw = kwargs.get("l1ot_lw", [0,1])
    inverse = kwargs.get("inverse_depth", False)
    ot_iter = kwargs.get("ot_iter", 3)
    ot_eps = kwargs.get("ot_eps", 1)
    ot_continous = kwargs.get("ot_continous", False)
    depth_max = kwargs.get("depth_max", 100)
    depth_min = kwargs.get("depth_min", 1)
    mono = kwargs.get("mono", False)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    stage_ot_loss = []
    stage_l1_loss = []
    range_err_ratio = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        depth_pred = stage_inputs['depth']
        hypo_depth = stage_inputs['hypo_depth']
        attn_weight = stage_inputs['attn_weight']
        B,H,W = depth_pred.shape
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        depth_gt = depth_gt_ms[stage_key]
        depth_pred_norm = depth_pred * 128 / (depth_max - depth_min)[:,None,None]  # B H W
        depth_gt_norm = depth_gt * 128 / (depth_max - depth_min)[:,None,None]  # B H W

        if mono and stage_idx!=0:
            this_stage_l1_loss = F.l1_loss(stage_inputs['mono_depth'][mask], depth_gt[mask], reduction='mean')
        else:
            this_stage_l1_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

        if inverse:
            depth_itv = (1/hypo_depth[:,2,:,:]-1/hypo_depth[:,1,:,:]).abs()  # B H W
            mask_out_of_range = ((1/hypo_depth - 1/depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0 # B H W
        else:
            depth_itv = (hypo_depth[:,2,:,:]-hypo_depth[:,1,:,:]).abs()  # B H W
            mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0 # B H W
        range_err_ratio.append(mask_out_of_range[mask].float().mean())

        this_stage_ot_loss = sinkhorn(depth_gt, hypo_depth, attn_weight, mask, iters=ot_iter, eps=ot_eps, continuous=ot_continous)[1]

        stage_l1_loss.append(this_stage_l1_loss)
        stage_ot_loss.append(this_stage_ot_loss)
        total_loss = total_loss + stage_lw[stage_idx] * (l1ot_lw[0] * this_stage_l1_loss + l1ot_lw[1] * this_stage_ot_loss)

    abs_err = torch.abs(depth_pred_norm[mask] - depth_gt_norm[mask])
    epe = abs_err.mean()
    err3 = (abs_err<=3).float().mean()*100
    err1= (abs_err<=1).float().mean()*100
    return total_loss, stage_l1_loss, stage_ot_loss, range_err_ratio, epe, err3, err1