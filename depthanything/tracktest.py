import cv2
import torch
import numpy as np 
from depth_anything_v2.dpt import DepthAnythingV2
import os

def run_depth_anything_v2(encoder): 
    '''
        encoder: 
            `vits`
            `vitb`
            `vitl`
            `vitg`
    '''
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything_path = r"C:\Users\arthu\Depth-Anything-V2\checkpoints"
    checkpoint_path = os.path.join(depth_anything_path, f"depth_anything_v2_{encoder}.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    capObj = cv2.VideoCapture(0)

    if not capObj.isOpened():
        print("Error: Could not access the camera.")
        exit()

    while True: 
        ret,frame = capObj.read()

        if not ret or frame is None:
            print("Error: Could not read frame from camera.")
            break  

        depthMap = model.infer_image(frame) 
        depthMap_normalized = cv2.normalize(depthMap, None, 0, 255, cv2.NORM_MINMAX)
        depthMap_8bit = np.uint8(depthMap_normalized)
        depthMap_inferno = cv2.applyColorMap(depthMap_8bit, cv2.COLORMAP_INFERNO)
        combined = np.hstack((frame, depthMap_inferno))
        cv2.imshow('Combined', combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capObj.release()
    cv2.destroyAllWindows()

if __name__ == '__main__': 
    run_depth_anything_v2('vits')