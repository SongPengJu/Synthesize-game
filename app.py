import cv2
import mediapipe as mp
import numpy as np
import math
import random
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- 1. 配置参数 (保持你的设置) ---
FRUIT_CONFIG = [
    {'r': 22,  'c': (120, 200, 120), 'name': 'Grape'},      
    {'r': 32,  'c': (80, 80, 225),   'name': 'Cherry'},     
    {'r': 46,  'c': (50, 220, 240),  'name': 'Lemon'},      
    {'r': 60,  'c': (80, 160, 255),  'name': 'Orange'},     
    {'r': 72,  'c': (100, 100, 255), 'name': 'Peach'},      
    {'r': 90,  'c': (80, 200, 50),   'name': 'Melon'},      
    {'r': 110, 'c': (240, 240, 240), 'name': 'Coconut'},    
    {'r': 140, 'c': (50, 215, 255),  'name': 'HAPPY KING'} 
]

# 物理参数
GRAVITY = 0.35          
FRICTION = 0.96         
AIR_RESISTANCE = 0.96   
BOUNCINESS = 0.4        
PHYSICS_SUBSTEPS = 8    

# 字体配置
FONT_MAIN = cv2.FONT_HERSHEY_SIMPLEX

# --- 2. 辅助类与函数 ---

class SpriteLoader:
    def __init__(self):
        self.sprites = {}

    def load_sprites(self):
        # 网页版适当调整渲染倍率保证性能
        SCALE = 1.0 
        for i, cfg in enumerate(FRUIT_CONFIG):
            r = int(cfg['r'] * SCALE)
            d = r * 2
            img = np.zeros((d, d, 4), dtype=np.uint8)
            center = (r, r)
            # 主体
            cv2.circle(img, center, r-2, cfg['c'], -1, cv2.LINE_AA)
            cv2.circle(img, center, r-2, (255, 255, 255), 2, cv2.LINE_AA)
            

            if i == 7: 
                # === Level 7: HAPPY KING ===
                # B. 眼睛 (开心的弯弯眼)
                eye_color = (50, 50, 50)
                eye_w = r // 5
                cv2.ellipse(img, (r - r//3, r - r//6), (eye_w, eye_w), 0, 180, 360, eye_color, 4, cv2.LINE_AA)
                cv2.ellipse(img, (r + r//3, r - r//6), (eye_w, eye_w), 0, 180, 360, eye_color, 4, cv2.LINE_AA)
                # C. 嘴巴 (张开大笑 D形)
                mouth_y = r + r//5
                mouth_w = r // 3
                mouth_h = r // 4
                cv2.ellipse(img, (r, mouth_y), (mouth_w, mouth_h), 0, 0, 180, (50, 50, 200), -1, cv2.LINE_AA) # 内部
                cv2.ellipse(img, (r, mouth_y), (mouth_w, mouth_h), 0, 0, 180, eye_color, 3, cv2.LINE_AA) # 边框
            else:
                # 普通表情
                # 高光
                cv2.circle(img, (r - r//3, r - r//3), r//4, (255, 255, 255, 180), -1, cv2.LINE_AA)
                eye_y = r + r//5
                cv2.circle(img, (r - r//3, eye_y), max(2, r//10), (50,50,50), -1, cv2.LINE_AA)
                cv2.circle(img, (r + r//3, eye_y), max(2, r//10), (50,50,50), -1, cv2.LINE_AA)
                cv2.ellipse(img, (r, eye_y + r//6), (r//5, r//8), 0, 0, 180, (50,50,50), 2, cv2.LINE_AA)

            # Mask
            mask = np.zeros((d, d), dtype=np.uint8)
            cv2.circle(mask, center, r-2, 255, -1)
            img[:, :, 3] = mask
            self.sprites[i] = img
        return self.sprites

def rotate_image(image, angle):
    h, w = image.shape[:2]
    image_center = (w//2, h//2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
    return result

def overlay_image_alpha(img, img_overlay, pos):
    x, y = pos
    h, w = img.shape[:2]
    h_o, w_o = img_overlay.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w_o, w), min(y + h_o, h)
    if x1 >= x2 or y1 >= y2: return img
    
    roi = img[y1:y2, x1:x2]
    overlay_crop = img_overlay[(y1-y):(y2-y), (x1-x):(x2-x)]
    
    alpha = overlay_crop[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha
    
    for c in range(3):
        roi[:, :, c] = (alpha * overlay_crop[:, :, c] + alpha_inv * roi[:, :, c])
    return img

class PhysObject:
    def __init__(self, x, y, level):
        self.x = x
        self.y = y
        self.level = level
        self.radius = FRUIT_CONFIG[level]['r']
        self.mass = self.radius * self.radius 
        self.vx = 0
        self.vy = 0
        self.angle = 0 
        self.is_static = False
        self.to_delete = False

    def update(self, box_w, box_h):
        if self.is_static: return
        self.vy += GRAVITY
        self.vx *= AIR_RESISTANCE
        self.vy *= AIR_RESISTANCE
        self.x += self.vx
        self.y += self.vy
        self.angle -= (self.vx / self.radius) * 180 / math.pi 

        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx * BOUNCINESS
            self.vy *= FRICTION
        elif self.x + self.radius > box_w:
            self.x = box_w - self.radius
            self.vx = -self.vx * BOUNCINESS
            self.vy *= FRICTION
        if self.y + self.radius > box_h:
            self.y = box_h - self.radius
            self.vy = -self.vy * BOUNCINESS
            self.vx *= FRICTION 
            if abs(self.vy) < GRAVITY * 2: self.vy = 0
            if abs(self.vx) < 0.1: self.vx = 0

    def draw(self, img, sprites, offset_x, offset_y):
        sprite = sprites[self.level]
        draw_sprite = sprite
        if abs(self.angle) > 1 or abs(self.vx) > 0.1:
            draw_sprite = rotate_image(sprite, self.angle)
        top_left_x = int(self.x + offset_x - self.radius)
        top_left_y = int(self.y + offset_y - self.radius)
        overlay_image_alpha(img, draw_sprite, (top_left_x, top_left_y))

def resolve_collisions(objects, score):
    new_objects = []
    for i in range(len(objects)):
        obj1 = objects[i]
        if obj1.to_delete: continue
        for j in range(i + 1, len(objects)):
            obj2 = objects[j]
            if obj2.to_delete: continue
            
            dx = obj2.x - obj1.x
            dy = obj2.y - obj1.y
            dist_sq = dx*dx + dy*dy
            min_dist = obj1.radius + obj2.radius
            
            if dist_sq < min_dist * min_dist:
                dist = math.sqrt(dist_sq)
                if dist == 0: dist, dx, dy = 0.1, 0.1, 0
                
                # 合成逻辑
                if obj1.level == obj2.level and obj1.level < 7:
                    obj1.to_delete = True
                    obj2.to_delete = True
                    mx, my = (obj1.x+obj2.x)/2, (obj1.y+obj2.y)/2
                    new_obj = PhysObject(mx, my, obj1.level + 1)
                    new_obj.vy = -2
                    new_obj.vx = random.uniform(-1, 1)
                    new_objects.append(new_obj)
                    score += (obj1.level + 1) * 20
                else:
                    # 碰撞逻辑
                    overlap = min_dist - dist
                    nx, ny = dx / dist, dy / dist
                    total_mass = obj1.mass + obj2.mass
                    r1 = obj2.mass / total_mass
                    r2 = obj1.mass / total_mass
                    
                    obj1.x -= nx * overlap * r1
                    obj1.y -= ny * overlap * r1
                    obj2.x += nx * overlap * r2
                    obj2.y += ny * overlap * r2
                    
                    rvx = obj2.vx - obj1.vx
                    rvy = obj2.vy - obj1.vy
                    vel_normal = rvx * nx + rvy * ny
                    if vel_normal > 0: continue
                    
                    restitution = 0.2
                    j = -(1 + restitution) * vel_normal
                    j /= (1/obj1.mass + 1/obj2.mass)
                    impulse_x = j * nx
                    impulse_y = j * ny
                    
                    obj1.vx -= impulse_x / obj1.mass
                    obj1.vy -= impulse_y / obj1.mass
                    obj2.vx += impulse_x / obj2.mass
                    obj2.vy += impulse_y / obj2.mass
                    
    objects = [o for o in objects if not o.to_delete]
    objects.extend(new_objects)
    return objects, score

# --- 3. 核心游戏处理器 (WebRTC 版) ---
class GameProcessor(VideoTransformerBase):
    def __init__(self):
        # 初始化 MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
        
        # 初始化素材
        loader = SpriteLoader()
        self.sprites = loader.load_sprites()
        
        # 游戏状态
        self.game_objects = []
        self.score = 0
        self.holding_obj = None
        self.next_level = 0
        self.spawn_timer = 0
        self.game_over = False
        self.danger_timer = 0.0
        
        # 技能状态
        self.shake_cooldown = 0
        self.shake_visual_offset = (0, 0)
        self.shake_visual_timer = 0

    def detect_fist(self, landmarks):
        wrist = landmarks[0]
        tips = [8, 12, 16, 20]
        folded_count = 0
        for t in tips:
            tip = landmarks[t]
            dist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
            if dist < 0.25: folded_count += 1
        return folded_count >= 3 

    def draw_glass_panel(self, img, x, y, w, h):
        # 震动视觉偏移
        dx, dy = self.shake_visual_offset
        x += dx
        y += dy
        
        # 绘制玻璃背景
        sub = img[y:y+h, x:x+w]
        # 模糊处理
        blur = cv2.GaussianBlur(sub, (25, 25), 0)
        white = np.full(blur.shape, 255, dtype=np.uint8)
        cv2.addWeighted(blur, 0.7, white, 0.3, 0, sub)
        img[y:y+h, x:x+w] = sub
        
        # 边框颜色
        border_color = (0, 215, 255) if self.shake_cooldown == 0 else (200, 200, 200)
        cv2.rectangle(img, (x, y), (x+w, y+h), border_color, 2)
        return x, y 

    def recv(self, frame):
        """
        Streamlit WebRTC 的核心回调函数。
        """
        # 1. 转换图像
        img = frame.to_ndarray(format="bgr24")
        # 强制将图像缩放到 1280x720，防止因分辨率变化导致球体大小比例失调
        # 同时也保证了物理引擎（重力、速度）在不同设备上的手感一致
        img = cv2.resize(img, (1280, 720))
        img = cv2.flip(img, 1)
        H, W = img.shape[:2] 
        
        # 2. 布局参数
        BOX_W = int(W * 0.55)
        BOX_H = int(H * 0.85)
        OFF_X = (W - BOX_W) // 2
        OFF_Y = (H - BOX_H) // 2
        DANGER_Y = OFF_Y + int(BOX_H * 0.15)

        # 3. 手势识别
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        
        is_pinching = False
        is_fist = False
        hand_x = 0
        
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            x4, y4 = int(lm.landmark[4].x * W), int(lm.landmark[4].y * H)
            x8, y8 = int(lm.landmark[8].x * W), int(lm.landmark[8].y * H)
            cx, cy = (x4+x8)//2, (y4+y8)//2
            
            # 捏合检测
            if math.hypot(x4-x8, y4-y8) < 40: is_pinching = True
            
            # 握拳检测
            is_fist = self.detect_fist(lm.landmark)
            hand_x = cx

            # 绘制光标
            if is_fist:
                cv2.putText(img, "FIST!", (cx, cy-20), FONT_MAIN, 1, (0,0,255), 2)
                cv2.circle(img, (cx, cy), 15, (0, 0, 255), -1)
            else:
                color = (0, 255, 0) if is_pinching else (0, 100, 255)
                cv2.circle(img, (cx, cy), 8, color, -1)

        # 4. 游戏逻辑
        if not self.game_over:
            # === 技能逻辑 ===
            if is_fist and self.shake_cooldown == 0:
                self.shake_cooldown = 90
                self.shake_visual_timer = 15
                for obj in self.game_objects:
                    if not obj.is_static:
                        obj.vy -= random.uniform(5, 12)
                        obj.vx += random.uniform(-10, 10)
            
            if self.shake_cooldown > 0: self.shake_cooldown -= 1
            if self.shake_visual_timer > 0:
                self.shake_visual_timer -= 1
                self.shake_visual_offset = (random.randint(-10, 10), random.randint(-5, 5))
            else:
                self.shake_visual_offset = (0, 0)

            # === 绘制场景 ===
            draw_x, draw_y = self.draw_glass_panel(img, OFF_X, OFF_Y, BOX_W, BOX_H)
            
            # 危险线
            cv2.line(img, (OFF_X, DANGER_Y), (OFF_X + BOX_W, DANGER_Y), (150, 150, 150), 2)
            
            is_danger = False
            for obj in self.game_objects:
                if obj.y < (DANGER_Y - OFF_Y) and not obj.is_static:
                    is_danger = True
                    break
            if is_danger:
                self.danger_timer += 1/30.0 # 假设 30FPS
                prog = min(self.danger_timer / 2.0, 1.0)
                cv2.rectangle(img, (OFF_X, DANGER_Y-10), (int(OFF_X + BOX_W*prog), DANGER_Y), (0,0,255), -1)
                if self.danger_timer >= 2.0: self.game_over = True
            else:
                self.danger_timer = 0

            # === 生成与抓取 ===
            rel_hand_x = hand_x - OFF_X
            if self.holding_obj is None:
                if is_pinching and not is_fist and self.spawn_timer == 0 and self.shake_visual_timer == 0:
                    spawn_x = max(20, min(BOX_W-20, rel_hand_x))
                    self.holding_obj = PhysObject(spawn_x, 40, self.next_level)
                    self.holding_obj.is_static = True
            else:
                target_x = max(self.holding_obj.radius, min(BOX_W-self.holding_obj.radius, rel_hand_x))
                self.holding_obj.x = target_x
                self.holding_obj.y = 40
                self.holding_obj.draw(img, self.sprites, draw_x, draw_y)
                cv2.line(img, (int(target_x+draw_x), draw_y+40), (int(target_x+draw_x), draw_y+BOX_H), (200,200,200), 1)
                
                if not is_pinching:
                    self.holding_obj.is_static = False
                    self.game_objects.append(self.holding_obj)
                    self.holding_obj = None
                    self.next_level = random.randint(0, 3)
                    self.spawn_timer = 15

            if self.spawn_timer > 0: self.spawn_timer -= 1
            
            # === 物理更新 ===
            for _ in range(PHYSICS_SUBSTEPS):
                for obj in self.game_objects:
                    obj.update(BOX_W, BOX_H)
                self.game_objects, self.score = resolve_collisions(self.game_objects, self.score)
            
            # 绘制物体
            for obj in self.game_objects:
                obj.draw(img, self.sprites, draw_x, draw_y)

            # === UI 信息 ===
            if self.shake_cooldown == 0:
                cv2.putText(img, "SKILL READY! (FIST)", (OFF_X, OFF_Y + BOX_H + 30), FONT_MAIN, 0.8, (0, 215, 255), 2)
            else:
                cv2.putText(img, "RECHARGING...", (OFF_X, OFF_Y + BOX_H + 30), FONT_MAIN, 0.8, (100, 100, 100), 2)

            cv2.putText(img, f"{self.score}", (OFF_X, OFF_Y - 10), FONT_MAIN, 1, (255,255,255), 2)
            
            # Next Fruit
            next_p = cv2.resize(self.sprites[self.next_level], (40, 40))
            overlay_image_alpha(img, next_p, (OFF_X + BOX_W - 50, OFF_Y - 50))

        else:
            # === GAME OVER ===
            overlay = np.zeros_like(img)
            cv2.addWeighted(img, 0.4, overlay, 0.6, 0, img)
            cv2.putText(img, "GAME OVER", (W//2 - 180, H//2), FONT_MAIN, 2.5, (255, 255, 255), 4)
            cv2.putText(img, "Pinch to Restart", (W//2 - 110, H//2 + 80), FONT_MAIN, 0.8, (255, 255, 255), 1)
            
            # 重启检测
            if is_pinching and self.spawn_timer == 0:
                self.game_objects = []
                self.score = 0
                self.holding_obj = None
                self.next_level = 0
                self.spawn_timer = 30
                self.game_over = False

        # 返回视频帧给 Streamlit 显示
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. 网页入口 ---
st.set_page_config(page_title="Synthesize a good mood", page_icon="😁", layout="wide")

st.title("😁 Synthesize a good mood")
st.write("请点击下方 **START** 并允许摄像头权限。")

webrtc_streamer(
    key="Synthesize-game",
    video_processor_factory=GameProcessor,
    rtc_configuration={ 
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": {"width": 1280, "height": 720}, # 请求高清分辨率
        "audio": False
    },
    mode=WebRtcMode.SENDRECV,
    async_processing=True,
)