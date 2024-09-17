from enum import Enum
import gymnasium as gym
import numpy as np
import mss
from PIL import Image
import easyocr
import time
import pydirectinput
# from pydirectinput import moveRel
import torch
import vgamepad as vg
import warnings
import cv2
import sys
import warnings
from torch.cuda.amp import autocast

# warnings.filterwarnings("ignore", category=FutureWarning, module="easyocr")
warnings.filterwarnings("ignore", category=FutureWarning)
# gamepad = vg.VX360Gamepad()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

N_CHANNELS = 1
HEIGHT = 1080
WIDTH = 1920

RESIZE_FACTOR = 3
pydirectinput.FAILSAFE = False
class DetectionState(Enum):
    DETECTED_TARGET = 1
    DETECTED_OTHER = 2
    DETECTED_NOTHING = 3



class FortniteEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Define continuous action space: [right_thumb_x, right_thumb_y, trigger]
        trigger = 1
        actions = trigger
        self.action_space = gym.spaces.Dict({
            'fire': gym.spaces.Discrete(actions),
            # 'look': gym.spaces.Box(low=-1, high=1, shape=(0,), dtype=np.float32)
        })
        # self.action_space = gym.spaces.Discrete(2)
        
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                            shape=(HEIGHT//RESIZE_FACTOR, WIDTH//RESIZE_FACTOR, N_CHANNELS), dtype=np.uint8)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = self.load_model()
        self.cam = mss.mss()
        self.batch_size = torch.Size([1])
        self.frame_stack = 4
    
        top_left_x, top_left_y = 500, 190
        bottom_right_x, bottom_right_y = 790, 300

        width = bottom_right_x - top_left_x
        height = bottom_right_y - top_left_y
        self.monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        self.kill_monitor = {"top": top_left_y, "left": top_left_x, "width": width, "height": height}

        top_left_x, top_left_y = 145,40
        bottom_right_x, bottom_right_y = 400,90
        width = bottom_right_x - top_left_x
        height = bottom_right_y - top_left_y
        self.check_game = {"top": top_left_y, "left": top_left_x, "width": width, "height": height}
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        self.end_game_if_not_game = 0
        self.cur_step = 0
        self.step_since_last_reset = 0
        self.score = 0
        self.score_detected_cooldown_period = False

        self.aiming_time = 0
        self.last_position = None
        self.stationary_time = 0
        self.max_aiming_time = 2
        self.max_stationary_time = 10

        # Reward system constants
        self.base_reward = -1
        self.aiming_reward = 1
        self.aiming_penalty = -10
        self.kill_reward = 50
        self.firing_reward = 1


    def step(self, action):
        trigger = action['fire']
        # look = action['look']
        # trigger = -1
        # right_thumb_x = int(right_thumb_x*20)

        # pydirectinput.smooth_move(right_thumb_x, 0, duration=0.1,steps=50)
        if trigger > 0:
            pydirectinput.mouseDown(button='left')
            pydirectinput.mouseUp(button='left')


        player_obs = None
        enemies = []
        center_x, center_y = 0, 0
        score_detected = DetectionState.DETECTED_NOTHING

        # try:
        player_full_img = np.array(self.cam.grab(self.monitor))
        kill_full_img = np.array(self.cam.grab(self.kill_monitor))
        check_game = np.array(self.cam.grab(self.check_game))

        player_obs = self.quarter_sized_screencap_np(player_full_img)
        enemies, center_x, center_y = self.detect_enemies(player_full_img)
        score_detected = self.score_detected(kill_full_img)
        game_running = self.chech_game_runnig(check_game)
        # except Exception as e:
        #         print(f"step {self.cur_step} screencap or processing fail {e}")

        if not game_running:
            self.end_game_if_not_game += 1
        else:
            self.end_game_if_not_game = 0
        reward = self.compute_reward(action, enemies, center_x, center_y, score_detected)
        # if reward > 0:
        #     image_path = rf"img_score/step_{self.cur_step}_score.png"
        #     cv2.imwrite(image_path, player_full_img)
        #     print(f"Saved state image as {image_path}")
        print(f"step {self.cur_step} trigger {trigger} reward {reward}")
        terminated = False
        truncated = False
        info = {}


        if player_obs is None:
            player_obs = np.zeros((HEIGHT//RESIZE_FACTOR, WIDTH//RESIZE_FACTOR, N_CHANNELS), dtype=np.uint8)

        self.cur_step += 1
        self.step_since_last_reset += 1
        if self.end_game_if_not_game > 20:
            return player_obs, 696969, True, truncated, info
    
        return player_obs, reward, terminated, truncated, info

        pass

    def reset(self, seed=None, options=None):

        try:
            player_obs = self.quarter_sized_screencap_np(np.array(self.cam.grab(self.monitor)))
        except:
            player_obs = np.zeros((HEIGHT//RESIZE_FACTOR, WIDTH//RESIZE_FACTOR, N_CHANNELS), dtype=np.uint8)
            print("reset screencap fail")
        self.step_since_last_reset = 0
        return player_obs, {}

    def close(self):
        pass
    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.to(self.device)
        return model
    
    def detect_enemies(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        center_x, center_y = frame_rgb.shape[1] // 2, frame_rgb.shape[0] // 2
        # with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        results = self.yolo_model(frame_rgb)
        enemies = []
        
        # Check if results is a list or tensor, and get the first item if it's a list
        for det in results.xyxy[0]:
            bbox = det[:4].cpu().numpy().astype(int)
            conf = float(det[4])
            cls = int(det[5])

            if cls == 0 and conf > 0.5:
                enemies.append(bbox)
        
        return enemies, center_x, center_y
    

    def quarter_sized_screencap_np(self, screencap_img):
        pil_image = Image.fromarray(screencap_img)
        gray_image = pil_image.resize((WIDTH//RESIZE_FACTOR, HEIGHT//RESIZE_FACTOR), Image.Resampling.LANCZOS).convert('L')
        return np.array(gray_image)[:, :, np.newaxis]

    def score_detected(self, full_img):
        elim_ocr = self.reader.readtext(full_img, detail=0)
        if len(elim_ocr) > 0:
            for text in elim_ocr:
                if '/' in text and '10,000' in text:
                    try:
                        kills, _ = text.split('/')
                        kills = kills.replace(',', '')
                        kills = int(kills)
                        if kills == 9999:
                            self.score = 0
                        if kills > self.score:
                            self.score = kills
                            print(f"score detected {kills}")
                            return DetectionState.DETECTED_TARGET
                    except ValueError:
                        print(f"score detected failed value error {text}")
                        continue
            return DetectionState.DETECTED_OTHER
        self.kill_count = 0
        return DetectionState.DETECTED_NOTHING
    def compute_reward(self, action, enemies, center_x, center_y, score_detected):
        # reward = self.base_reward
        reward = 1
        trigger = action['fire']
        # print(f"trigger {trigger}")
        # trigger = -1

        # Check if aiming at an enemy
        aiming_at_enemy = False
        if len(enemies) > 0:
            for enemy in enemies:
                x1, y1, x2, y2 = enemy
                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                    aiming_at_enemy = True
                    break
        # print(f"aiming at enemy {aiming_at_enemy}")
        if aiming_at_enemy:
            self.aiming_time += 1
            if self.aiming_time <= self.max_aiming_time:
                # if len(enemies) > 1:
                reward += 20*self.aiming_reward
                # print(f"step {self.cur_step} aiming at enemy reward {self.aiming_reward}")
            else:
                reward += self.aiming_penalty
                # print(f"step {self.cur_step} aiming too long, penalty applied")
        else:
            self.aiming_time = 0

        # Score detection reward
        if score_detected == DetectionState.DETECTED_TARGET:
            if trigger ==1:  # Firing at any value greater than 0
                firing_reward = self.firing_reward
                reward += firing_reward + self.kill_reward
                # print(f"step {self.cur_step} score detected and fired, reward {firing_reward + self.kill_reward}")
        else:
            if trigger ==1:
                reward -= 100*self.firing_reward  # Small penalty for firing without target
                # print(f"step {self.cur_step} firing without target, small penalty applied")
        return reward
    
    def chech_game_runnig(self,frame):
        emil_ocr = self.reader.readtext(frame, detail=0)
        # print(emil_ocr)
        if emil_ocr == ["GAME MODE"]:
            # print("GAME MODE")
            return True
        return False
        
    

    