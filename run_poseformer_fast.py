import os

import numpy as np
import torch

from common.arguments import parse_args
from common.camera import *
from common.model_poseformer import *


def eval_data_prepare(receptive_field, inputs_2d):
    inputs_2d_p = torch.squeeze(inputs_2d)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i, :, :, :] = inputs_2d_p[i : i + receptive_field, :, :]
    return eval_input_2d


if __name__ == "__main__":
    args = parse_args()

    keypoints = np.load(args.keypoints, allow_pickle=True)["keypoints"]
    keypoints = np.expand_dims(keypoints, axis=1)
    # TODO: fill if needed
    keypoints_metadata = None
    keypoints_symmetry = None

    width, height = 960, 540
    keypoints = normalize_screen_coordinates(keypoints, w=width, h=height)

    receptive_field = args.number_of_frames
    num_joints = 17
    # I copied these from run_poseformer.py
    # TODO: It should be checked if actually these correspond to our case
    kps_left = [4, 5, 6, 11, 12, 13]
    kps_right = [1, 2, 3, 14, 15, 16]
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    model_pos = PoseTransformerV2(
        num_frame=receptive_field,
        num_joints=num_joints,
        in_chans=2,
        num_heads=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_path_rate=0,
        args=args,
    )

    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print("Loading checkpoint", chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint["model_pos"], strict=False)

    inputs_2d = torch.from_numpy(keypoints.astype("float32"))
    ##### apply test-time-augmentation (following Videopose3d)
    inputs_2d_flip = inputs_2d.clone()
    inputs_2d_flip[:, :, :, 0] *= -1
    inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

    inputs_2d = eval_data_prepare(receptive_field, inputs_2d)
    inputs_2d_flip = eval_data_prepare(receptive_field, inputs_2d_flip)

    if torch.cuda.is_available():
        model_pos = nn.DataParallel(model_pos)
        model_pos = model_pos.cuda()
        inputs_2d = inputs_2d.cuda()
        inputs_2d_flip = inputs_2d_flip.cuda()

    predicted_3d_pos = model_pos(inputs_2d)
    print(f"{predicted_3d_pos.shape}")
    predicted_3d_pos_flip = model_pos(inputs_2d_flip)
    print(f"{predicted_3d_pos_flip.shape}")

    predicted_3d_pos_flip[:, :, :, 0] *= -1
    predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :, joints_right + joints_left]

    predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1, keepdim=True)

    prediction = predicted_3d_pos.squeeze(0).cpu().numpy()
    prediction = np.squeeze(prediction)

    print(f"Prediction shape: {prediction.shape}")

    # Next step: adapting the code for the visualization