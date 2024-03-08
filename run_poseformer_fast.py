import os

import numpy as np
import torch

from common.arguments import parse_args
from common.skeleton import Skeleton
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

    # nframes x 17 x
    keypoints = np.load(args.keypoints, allow_pickle=True)["keypoints"]

    # Pad sequence predict first and last frames
    receptive_field = args.number_of_frames
    pad = (receptive_field - 1) // 2  # Padding on each side
    keypoints = np.pad(keypoints, ((pad, pad), (0, 0), (0, 0)), "edge")

    # To match tensor dimensions in the original code
    keypoints = np.expand_dims(keypoints, axis=0)

    keypoints_metadata = {
        "layout_name": "h36m",
        "num_joints": 17,
        "keypoints_symmetry": [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]],
    }
    keypoints_symmetry = None
    azimuth = np.array(70.0, dtype="float32")
    n_frames = keypoints.shape[0]

    h36m_skeleton = Skeleton(
        parents=[
            -1,
            0,
            1,
            2,
            3,
            4,
            0,
            6,
            7,
            8,
            9,
            0,
            11,
            12,
            13,
            14,
            12,
            16,
            17,
            18,
            19,
            20,
            19,
            22,
            12,
            24,
            25,
            26,
            27,
            28,
            27,
            30,
        ],
        joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
        joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31],
    )

    h36m_skeleton.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
    h36m_skeleton._parents[11] = 8
    h36m_skeleton._parents[14] = 8

    width, height = 960, 540
    keypoints = normalize_screen_coordinates(keypoints, w=width, h=height)

    num_joints = 17

    # I copied these from run_poseformer.py
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

    if torch.cuda.is_available():
        model_pos = nn.DataParallel(model_pos)
        model_pos = model_pos.cuda()

    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print("Loading checkpoint", chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint["model_pos"], strict=False)

    with torch.no_grad():
        inputs_2d = torch.from_numpy(keypoints.astype("float32"))
        ##### apply test-time-augmentation (following Videopose3d)
        inputs_2d_flip = inputs_2d.clone()
        inputs_2d_flip[:, :, :, 0] *= -1
        inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

        inputs_2d = eval_data_prepare(receptive_field, inputs_2d)
        inputs_2d_flip = eval_data_prepare(receptive_field, inputs_2d_flip)

        if torch.cuda.is_available():
            inputs_2d = inputs_2d.cuda()
            inputs_2d_flip = inputs_2d_flip.cuda()

        predicted_3d_pos = model_pos(inputs_2d)
        predicted_3d_pos_flip = model_pos(inputs_2d_flip)

        predicted_3d_pos_flip[:, :, :, 0] *= -1
        predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[
            :, :, joints_right + joints_left
        ]

        predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1, keepdim=True)

        prediction = predicted_3d_pos.squeeze(0).cpu().numpy()

    prediction = np.squeeze(prediction)

    print(f"Prediction shape: {prediction.shape}")

    if args.save_output:
        kpts_filename = os.path.basename(args.keypoints)
        np.savez(os.path.join(args.model_output, "3d_" + kpts_filename), keypoints=prediction)

    if args.render:
        # For visualization purposes, make all join coordinates relative to the hip joint,
        # being the hip joint the (0, 0, 0). This is because we don't have the camera extrinsics.
        prediction -= np.expand_dims(prediction[:, 0, :], axis=1)

        # Hardcoded (experimentally found) rotation matrix. Only for visualization purposes.
        angle_x = 60 * (np.pi / 180)
        angle_y = 180 * (np.pi / 180)
        angle_z = -30 * (np.pi / 180)

        rot_x = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
        rot_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
        rot_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0], [np.sin(angle_z), np.cos(angle_z), 0], [0, 0, 1]])

        # Hardcoded (experimentally found) translation
        tx = 0
        ty = 0
        tz = 0.75
        t = np.array([tx, ty, tz])

        # Apply extrinsics to predictions
        prediction = (prediction @ rot_x.T @ rot_y.T @ rot_z.T) + t

        anim_output = {"Reconstruction": prediction}
        keypoints = image_coordinates(keypoints, w=width, h=height)
        keypoints = np.squeeze(keypoints)

        # Create visualization
        from common.visualization import render_animation

        # Remove the padding
        keypoints = keypoints[pad:-pad]

        render_animation(
            keypoints,
            keypoints_metadata,
            anim_output,
            h36m_skeleton,
            15,
            args.viz_bitrate,
            azimuth,
            args.viz_output,
            limit=args.viz_limit,
            downsample=args.viz_downsample,
            size=args.viz_size,
            input_video_path=args.viz_video,
            viewport=(width, height),
            input_video_skip=args.viz_skip,
        )
