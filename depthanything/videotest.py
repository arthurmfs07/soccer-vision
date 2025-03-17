import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--img-path', type=str, default=r"C:\Users\arthu\viscompPCD\soccer-vision-1\depthanything\vid\basketball.mp4")
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default=os.path.join(os.path.dirname(__file__), 'picsout'))

    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])

    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    checkpoint_dir = r"C:\Users\arthu\Depth-Anything-V2\checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, f"depth_anything_v2_{args.encoder}.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Open video file
    cap = cv2.VideoCapture(args.img_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.img_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the output video path
    out_video_path = os.path.join(args.outdir, 'output_video.mp4')

    # Use XVID codec or another suitable one
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_width * 2 + 50, frame_height))

    # Ensure the output directory exists
    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame {frame_idx + 1}")
        frame_idx += 1

        # Run depth estimation
        depth = depth_anything.infer_image(frame, args.input_size)

        # Normalize depth to 0-255
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        if args.pred_only:
            # Write depth image only
            cv2.imwrite(os.path.join(args.outdir, f"frame_{frame_idx:04d}.png"), depth)
        else:
            # Combine raw frame and depth map
            split_region = np.ones((frame.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([frame, split_region, depth])

            # Write combined result to video
            out_video.write(combined_result)

    # Release video objects
    cap.release()
    out_video.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    # Verify if the video is saved and openable
    cap_test = cv2.VideoCapture(out_video_path)
    if not cap_test.isOpened():
        print(f"Failed to open the saved video: {out_video_path}")
    else:
        print(f"Successfully saved and opened the video: {out_video_path}")
    cap_test.release()
