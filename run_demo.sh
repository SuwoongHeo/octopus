#!/usr/bin/env bash
#python infer_single.py sample data/sample/segmentations data/sample/keypoints --out_dir out
dataroot=/ssd2/swheo/dev/HumanRecon_/
view=4
datapath=${dataroot}data/SampleBEJ/4view_nocalib/4view_input_0

#name=SWH
#datapath=data/Sample_HJH/nview/4view_input
name=BEJ
echo ${datapath}/segmentations
python infer_single.py --name ${name}${view}_p40_s40 --segm_dir ${datapath}/segmentations --pose_dir ${datapath}/keypoints --out_dir ${datapath}/out -s 40 -p 40
python infer_single.py --name ${name}${view}_p20_s20 --segm_dir ${datapath}/segmentations --pose_dir ${datapath}/keypoints --out_dir ${datapath}/out -s 20 -p 20