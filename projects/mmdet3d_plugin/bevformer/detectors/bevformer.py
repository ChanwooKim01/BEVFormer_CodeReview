# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time


@DETECTORS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False, # True
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None, #
                 pts_backbone=None,
                 img_neck=None, #
                 pts_neck=None,
                 pts_bbox_head=None, #
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None, #
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False # True
                 ):

        super(BEVFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        
        # grid mask augmentation: 이미지를 일정한 간격으로 가리는 마스크를 씌워주는 것 (검은색으로 채워버림!)
        # use_h, use_w, rotate, offset, ratio, mode, prob
        # use_h, use_w: 이미지의 높이, 너비에 대해 마스크를 씌울 것인지
        # rotate: 마스크를 회전시킬 각도
        # offset: 마스크를 씌운 후, 이미지에 랜덤한 offset을 더해줄 것인지
        # ratio: 마스크의 비율
        # mode: 1이면 마스킹 후 1-mask를 반환, 즉 마스크를 씌운 부분만 남기고 나머지를 검은색으로 채움
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    # 이미지 피처 추출: ResNet -> FPN
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size() # 아마 N은 이미지 수 즉 6개이겠지?
                img = img.reshape(B * N, C, H, W)
            
            # grid mask augmentation: 이미지를 일정한 간격으로 가리는 마스크를 씌워주는 것 (검은색으로 채워버림!)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            # ResNet에서 이미지 피쳐 추출
            img_feats = self.img_backbone(img)
            # ResNet 모델 아웃풋이 딕셔너리일 경우, 리스트로 변환
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        # FPN 넥에서 이미지 피쳐 추출
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        # 이미지 피쳐를 큐 길이에 맞게 재배열
        # 즉, 이전 과정에서 ResNet 입력에 넣어주기 위해 (BN, C, H, W) 형태로 변환했던 것을 원래 형태로 변환해주는 과정인 듯
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats

    # BEVFormerHead
    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        # BEVFormerHead의 forward 메소드 호출
        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev) # only bev is false
        # output으로 출력된 prediction bbox 값: normalized된 값 -> sigmoid 역함수를 통해 정규화를 해제!
        
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        
        # loss cls: focal loss
        # loss bbox: L1 loss
        # loss iou: GIoU loss (fake cost)
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    # Temporal Self-Attention을 위한 과거 정보 추출 과정
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        # eval mode: gradients are not calculated
        self.eval()

        with torch.no_grad():
            prev_bev = None
            # 과거 이미지 자체를 저장하고 있는 듯? BEV 피처를 저장하는 것이 아니라
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            # 큐의 각 이미지에서 피처 추출
            # B, len_queue, num_cams, C, H, W로 차원 맞춰주기 위해 len_queue 도입
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue) 
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list] # len_queue번째 차원이라 생각하면 된다!
                
                # 각 과거 이미지에서의 BEV 피처 추출 
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            
            # 모든 과거 정보를 가져온 후엔 다시 train mode로 변환
            self.train()
            
            # 과거 BEV 피처 반환
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        len_queue = img.size(1) # 배치 안의 이미지 수
        prev_img = img[:, :-1, ...] # 과거 직전까지의 모든 이미지들
        img = img[:, -1, ...] # 현재 이미지

        prev_img_metas = copy.deepcopy(img_metas)
        
        # 위에서 본 함수! 과거 BEV 피처 추출
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue-1] for each in img_metas] #과거 BEV meta 정보?
        
        # prev bev가 존재하지 않는 경우 None으로 설정 (여기가 BEVFormer-S인가?)
        if not img_metas[0]['prev_bev_exists']: 
            prev_bev = None
            
        # 이미지 피처 추출: grid mask -> ResNet -> FPN
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        
        ############ BEVFormerHead의 forward 메소드 호출 (BEVFormerHead) ############
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)
        ###########################################################################
        
        losses.update(losses_pts)
        
        # loss 결과 반환
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            # img metas에 저장된 can_bus 정보를 이용해 ego motion을 계산 -> 이걸 다시 can bus에 저장?
            # 이전 씬에서의 pos와 angle을 빼줌으로써 가장 첫 번째 씬 기준에서 얼마나 pose가 변환되어 있는지 계산하는 듯
            # 이 부분이 alignment 작업인가? 라기보단 alignment를 계산하기 위한 값을 저장해주는 거 같기도...
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)

        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        
        # 반환하는 값은 bbox 결과만        
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        
        # BEVFormerHead
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)
        
        # BEVFormerHead의 get_bboxes 메소드 호출
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        
        # BEV 피처와 bbox 결과 반환
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        
        # BEVFormer의 simple_test_pts 메소드 호출
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        
        # 위에 선언한 비어있는 bbox_list에 bbox_pts를 넣어줌
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            
        # BEV 피처와 bbox 결과 반환    
        return new_prev_bev, bbox_list
