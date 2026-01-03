import sys, os
import argparse
import torch
import cv2
import numpy as np
from collections import OrderedDict

raft_root = os.path.abspath("RAFT")
sys.path.insert(0, raft_root)
sys.path.insert(0, os.path.join(raft_root, "core"))
sys.path.insert(0, os.path.join(raft_root, "utils"))

from core.raft import RAFT
from utils.utils import InputPadder


# Arguments RAFT 
parser = argparse.ArgumentParser() 
parser.add_argument('--small', action='store_true') 
parser.add_argument('--mixed_precision', action='store_true') 
parser.add_argument('--dropout', type=float, default=0.0) 
args = parser.parse_args([]) 
args.small = False   # mettre True si vous voulez RAFT-small 

# Charger modèle RAFT sur CPU 
device = "cpu" 
model = RAFT(args) 
state_dict = torch.load( 
os.path.join(raft_root, "models/raft-things.pth"), 
map_location=device 
) 

# Correction des clés ("module.") 
new_state_dict = OrderedDict() 
for k, v in state_dict.items(): 
    name = k[7:] if k.startswith("module.") else k 
    new_state_dict[name] = v 
model.load_state_dict(new_state_dict) 
model = model.to(device).eval() 

# Webcam 
cap = cv2.VideoCapture(0) 
ret, prev_frame = cap.read() 
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB) 
prev_frame = torch.from_numpy(prev_frame).float() / 255.0 
prev_frame = prev_frame.permute(2, 0, 1)[None].to(device) 

while True: 
    ret, frame = cap.read() 
    if not ret: 
        break 
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0 
    frame_tensor = frame_tensor.permute(2, 0, 1)[None].to(device)
    
    # Padding pour chaque paire d'images 
    padder = InputPadder(prev_frame.shape) 
    prev_pad, frame_pad = padder.pad(prev_frame, frame_tensor) 
    with torch.no_grad(): 
        _, flow_up = model(prev_pad, frame_pad, iters=12, test_mode=True) 
    
    # iters=12 pour accélérer sur CPU 
    flow = flow_up[0].permute(1, 2, 0).cpu().numpy() 

    # Visualisation HSV 
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8) 
    hsv[..., 0] = ang * 180 / np.pi / 2 
    hsv[..., 1] = 255 
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) 
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 
    cv2.imshow("RAFT Optical Flow (CPU)", bgr) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 

prev_frame = frame_tensor 
cap.release() 
cv2.destroyAllWindows() 