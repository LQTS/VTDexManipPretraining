import os.path
import pickle

import cv2
# from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

from typing import List
import numpy
import numpy as np
from pathlib import Path

def match_timestamp(target, all_timestamps):
    return np.argmin([abs(x - target) for x in all_timestamps])


def preprocess_tactile_binary(
    threshold: float,
    tactile_data: List[List[List[float]]]
) -> List[List[List[int]]]:
    tactile_data = np.array(tactile_data)
    tactile_1_idx = tactile_data > threshold
    tactile_data[tactile_1_idx] = 1
    tactile_data[~tactile_1_idx] = 0

    return tactile_data.tolist()




def visualize_in_generation(v_path, Hand_Pic_Path, touch_data):


    touch_data = preprocess_tactile_binary(0.2, touch_data)
    touch_data = np.array(touch_data)


    image_files = [f for f in os.listdir(v_path) if f.endswith('.png')]
    image_files = sorted(image_files, key=lambda s: int(s.split('=')[-1].split('.')[0]))
    video, output_video_path = init_vedio(image_files, v_path, v_path.parent.parent.parent)


    hand_image = cv2.imread(Hand_Pic_Path, cv2.IMREAD_UNCHANGED)
    hand_image_gray = hand_image[:, :, 3]
    _, hand_image_gray = cv2.threshold(hand_image_gray, 5, 255, cv2.THRESH_BINARY_INV)


    # tv_images_dir = v_path.parent.parent.parent / 'visualize'
    # os.makedirs(tv_images_dir, exist_ok=True)

    print("begin to write video")
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(v_path, image_file)
        image = cv2.imread(image_path)

        # Add_Hand_Contour
        image = Add_Hand_Contour(image, hand_image_gray, [0, 0])

        draw_points(image, touch_data[i])
        # path = tv_images_dir / v_path.parts[-1]
        # os.makedirs(path, exist_ok=True)
        # cv2.imwrite(os.path.join(path, image_file), image)
        video.write(image)
    print(f'The video is saved in {output_video_path}')
    # release video
    video.release()

def Add_Hand_Contour(
    image: numpy.ndarray,
    hand_image: numpy.ndarray,
    hand_pos: List[int]
)->numpy.ndarray:
    img_width, img_height = image.shape[:2]
    hand_width, hand_height = hand_image.shape[:2]
    bottem_border_expand = img_height-hand_pos[1]-hand_height
    right_border_expand = img_width-hand_pos[0]-hand_width

    assert bottem_border_expand >= 0
    assert right_border_expand >= 0
    hand_image_mask = cv2.copyMakeBorder(hand_image,
                                         hand_pos[0], right_border_expand,
                                         hand_pos[1], bottem_border_expand,
                                         cv2.BORDER_CONSTANT, value=[255, 255, 255])


    return cv2.bitwise_and(image, image, mask=hand_image_mask)

def draw_points(image, torch_data):

    width, height = image.shape[:2]

    num_rows, num_cols = 4, 5


    # start_x = int((width - (num_cols * cell_size + (num_cols - 1) * spacing)) / 2)
    # start_y = int((height - (num_rows * cell_size + (num_rows - 1) * spacing)) / 2)

    sensor_pose_list = np.zeros((4, 5, 2))
    sensor_pose_list[0, 0] = [18, 55]
    sensor_pose_list[1, 0] = [28, 68]
    sensor_pose_list[2, 0] = [38, 78]
    sensor_pose_list[3, 0] = [48, 88]
    sensor_pose_list[0, 1] = [48, 30]
    sensor_pose_list[1, 1] = [54, 47]
    sensor_pose_list[2, 1] = [60, 64]
    sensor_pose_list[3, 1] = [66, 81]
    sensor_pose_list[0, 2] = [78, 15]
    sensor_pose_list[1, 2] = [80, 35]
    sensor_pose_list[2, 2] = [82, 55]
    sensor_pose_list[3, 2] = [84, 75]
    sensor_pose_list[0, 3] = [105, 22]
    sensor_pose_list[1, 3] = [106, 41]
    sensor_pose_list[2, 3] = [104, 60]
    sensor_pose_list[3, 3] = [103, 78]
    sensor_pose_list[0, 4] = [152, 66]
    sensor_pose_list[1, 4] = [141, 90]
    sensor_pose_list[2, 4] = [131, 100]
    sensor_pose_list[3, 4] = [70, 120]

    for row in range(num_rows):
        for col in range(num_cols):
            x = int(sensor_pose_list[row, col, 0])
            y = int(sensor_pose_list[row, col, 1])
            dot_color = calculate_touch_color(torch_data[row, col])
            cv2.circle(image, (x, y), 5, dot_color, -1, 2)



def calculate_touch_color(touch_data):

    h = 120 - touch_data * 120
    s = 80
    v = 70
    r, g, b = cv2.cvtColor(np.uint8([[[h / 2, (s / 100) * 255, (v / 100) * 255]]]), cv2.COLOR_HSV2RGB)[0][0]

    return (int(b), int(g), int(r))

def init_vedio(image_files, v_path, video_root_dir=None):
    if video_root_dir is None:
        output_video_path =  'visualize'
        os.makedirs(output_video_path, exist_ok=True)
        output_video = os.path.join(output_video_path, (v_path.split("/")[-2] + '.mp4'))
    else:
        output_video_path = video_root_dir / 'visualize'
        os.makedirs(output_video_path, exist_ok=True)
        output_video = os.path.join(output_video_path, (v_path.parts[-1] + '.mp4'))


    first_image = cv2.imread(os.path.join(v_path, image_files[0]))
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))
    return video, output_video_path


if __name__ == '__main__':

    Hand_Pic_Path = 'vitac_pretrain/hand.png'

    traj_id = '101000'


    ts_touch_data_path = 'data/raw/vt-dex-manip/tactile/' + traj_id
    v_data_path = Path('data/raw/vt-dex-manip/videos') / traj_id

    with open(ts_touch_data_path+'.pkl', 'rb') as f:
        touch_data = pickle.load(f)


    visualize_in_generation(v_data_path, Hand_Pic_Path, touch_data)