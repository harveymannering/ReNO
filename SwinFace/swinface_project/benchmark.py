import argparse

import cv2
import numpy as np
import torch

from model import build_model
import glob
import csv
import os

@torch.no_grad()
def inference(cfg, weight, img_path):
    if img_path is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    model = build_model(cfg)
    dict_checkpoint = torch.load(weight)
    model.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
    model.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
    model.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
    model.om.load_state_dict(dict_checkpoint["state_dict_om"])

    model.eval()
    output = model(img)#.numpy()

    #for each in output.keys():
    #    print(each, "\t" , output[each][0].numpy())
    return {
            'filename': os.path.basename(img_path),
            'face_detected': "?",
            'confidence': "?",
            'gender_preds': output['Gender'][0].tolist(),
            'gender': "Female" if np.argmax(output['Gender']) == 0 else "Male",
            'age': output['Age'][0].item(),
            'total_faces': "?",
            'error': None
        }

class SwinFaceCfg:
    network = "swin_t"
    fam_kernel_size=3
    fam_in_chans=2112
    fam_conv_shared=False
    fam_conv_mode="split"
    fam_channel_attention="CBAM"
    fam_spatial_attention=None
    fam_pooling="max"
    fam_la_num_list=[2 for j in range(11)]
    fam_feature="all"
    fam = "3x3_2112_F_s_C_N_max"
    embedding_size = 512

if __name__ == "__main__":

    cfg = SwinFaceCfg()
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--weight', type=str, default='<your path>/checkpoint_step_79999_gpu_0.pt')
    parser.add_argument('--directory', required=True, help='Directory containing images to process')
    parser.add_argument('--output', default='results_swin.csv', help='Output CSV file name')
    args = parser.parse_args()

    # Get all image files from directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.directory, extension.lower())))
        #image_files.extend(glob.glob(os.path.join(args.directory, extension.upper())))
    
    print(f"Found {len(image_files)} images to process...")
    
    # Process all images
    results = []
    for i, image_path in enumerate(image_files, 1):
        if int(image_path.split('_')[-1].split('.')[0]) <= 100:
            print(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
            result = inference(cfg, args.weight, image_path)
            if result:
                results.append(result)
                print(len(results))
    
    print("Final results length:", len(result))
    # Save results to CSV
    with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'face_detected', 'confidence', 'gender_preds', 'gender', 'age', 'total_faces', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\nResults saved to: {args.output}")
    
    # Print summary
    successful = sum(1 for r in results if r['face_detected'])
    failed = len(results) - successful
    print(f"Summary: {successful} images processed successfully, {failed} failed")
        
    
