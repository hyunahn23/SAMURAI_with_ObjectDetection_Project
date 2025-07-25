import cv2
import gc
import numpy as np
import os
import sys
import torch

from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO

def extract_frames_from_video(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"{frame_idx:05d}.jpg"), frame)
        frame_idx += 1
    cap.release()

def get_bbox_prompts():
    detection_model = YOLO(detection_ckpt)
    detection_results = detection_model(input_video)[0]
    bboxes = detection_results.boxes.xyxy
    bboxes_cpu = bboxes.to('cpu').numpy()
    prompts = {}
    for idx, bbox in enumerate(bboxes_cpu):
        prompts[idx] = ((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), 0)
    print(prompts)
    return prompts

if __name__=="__main__":
    detection_ckpt = "ckpts/yolo11x.pt"
    samurai_cfg = "configs/samurai/sam2.1_hiera_l.yaml"
    samurai_ckpt = "sam2/checkpoints/sam2.1_hiera_large.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    input_video = "data/test_video.mp4"  # YOLO 입력: mp4 파일
    frame_dir = "data/frames/"  # SAM2 입력: 프레임 디렉터리
    output_path = "result.mp4"

    # 프레임 추출
    extract_frames_from_video(input_video, frame_dir)

    samurai_predictor = build_sam2_video_predictor(samurai_cfg, samurai_ckpt, device=device)

    prompts = get_bbox_prompts()

    # Save
    video = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    loaded_video = [cv2.imread(video_path) for video_path in video]
    height, width = loaded_video[0].shape[:2]
    frame_rate = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        initial_state = samurai_predictor.init_state(frame_dir, offload_video_to_cpu=True)
        bbox, tracking_label = prompts[0]

        _, _, masks = samurai_predictor.add_new_points_or_box(initial_state, box=bbox, frame_idx=0, obj_id=0)

        for frame_idx, object_ids, masks in samurai_predictor.propagate_in_video(initial_state):
            mask_visualize = {}
            bbox_visualize = {}

            for object_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)

                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                bbox_visualize[object_id] = bbox
                mask_visualize[object_id] = mask

            # Visualize
            frame_img = loaded_video[frame_idx]
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 색상 다양화

            for object_id, mask in mask_visualize.items():
                mask_image = np.zeros((height, width, 3), np.uint8)
                mask_image[mask] = color[(object_id + 1) % len(color)]
                frame_img = cv2.addWeighted(frame_img, 1, mask_image, 0.7, 0)

            for object_id, bbox in bbox_visualize.items():
                cv2.rectangle(frame_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[object_id % len(color)], 2)

            output_video.write(frame_img)

        output_video.release()

    del samurai_predictor, initial_state
    gc.collect()
    torch.cuda.empty_cache()



# import cv2
# import gc
# import numpy as np
# import os
# import sys
# import torch
# # sys.path.append("./sam2")

# from sam2.build_sam import build_sam2_video_predictor
# from ultralytics import YOLO

# def get_bbox_prompts():
#     input_video = video_dir + "data/test_video.mp4"

#     detection_model = YOLO(detection_ckpt)
#     detection_results = detection_model(input_video)[0]

#     bboxes = detection_results.boxes.xyxy
#     bboxes_cpu = bboxes.to('cpu').numpy()

#     prompts = {}

#     for idx, bbox in enumerate(bboxes_cpu):
#         prompts[idx] = ((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), 0)
#     print(prompts)

#     return prompts

# if __name__=="__main__":
#     detection_ckpt = "ckpts/yolo11x.pt"
#     samurai_cfg = "configs/samurai/sam2.1_hiera_l.yaml"
#     samurai_ckpt = "sam2/checkpoints/sam2.1_hiera_large.pt"
#     device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

#     video_dir = "data/test_video.mp4"
#     output_path = "result.mp4"

#     samurai_predictor = build_sam2_video_predictor(samurai_cfg, samurai_ckpt, device=device)

#     #prompts = get_bbox_prompts(video_dir, detection_ckpt)
#     prompts = get_bbox_prompts()

#     # Save
#     video = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir)])
#     loaded_video = [cv2.imread(video_path) for video_path in video]
#     height, width = loaded_video[0].shape[:2]
#     frame_rate = 30

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     output_video = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

#     with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
#         initial_state = samurai_predictor.init_state(video_dir, offload_video_to_cpu=True)
#         bbox, tracking_label = prompts[0]

#         _, _, masks = samurai_predictor.add_new_points_or_box(initial_state, box=bbox, frame_idx=0, obj_id=0)

#         for frame_idx, object_ids, masks in samurai_predictor.propagate_in_video(initial_state):
#             mask_visualize = {}
#             bbox_visualize = {}

#             for object_id, mask in zip(object_ids, masks):
#                 mask = mask[0].cpu().numpy()
#                 mask = mask > 0.0
#                 non_zero_indices = np.argwhere(mask)

#                 if len(non_zero_indices) == 0: 
#                     bbox = [0, 0, 0, 0]
#                 else:
#                     y_min, x_min = non_zero_indices.min(axis=0).tolist()
#                     y_max, x_max = non_zero_indices.max(axis=0).tolist()
#                     bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                
#                 bbox_visualize[object_id] = bbox
#                 mask_visualize[object_id] = mask

#             # Visualize
#             frame_img = loaded_video[frame_idx]
#             color = [(255, 0, 0)]

#             for object_id, mask in mask_visualize.items():
#                 mask_image = np.zeros((height, width, 3), np.uint8)
#                 mask_image[mask] = color[(object_id + 1) % len(color)]

#                 frame_img = cv2.addWeighted(frame_img, 1, mask_image, 0.7, 0)

#             for object_id, bbox in bbox_visualize.items():
#                 cv2.rectangle(frame_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[object_id % len(color)], 2)

#             output_video.write(frame_img)

#         output_video.release()

#     del samurai_predictor, initial_state
#     gc.collect()
#     torch.clear_autocast_cache()
#     torch.cuda.empty_cache()