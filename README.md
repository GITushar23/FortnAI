# FortnAI
Watch the Demo
(https://img.youtube.com/vi/wSUpo9QFcRs/default.jpg)](https://youtu.be/wSUpo9QFcRs)

A reinforcement learning agent that learns directly from screen capture in Fortnite, implemented with DreamerV3 and computer vision.

## Overview

FortnAI creates an RL environment that learns from real-time screen capture instead of traditional simulation. The agent interacts with Fortnite through keyboard/mouse inputs, using YOLO for target detection.

## Technical Implementation

- **Custom RL Environment**: Real-time screen capture for state representation
- **DreamerV3**: Implemented from scratch in PyTorch 
- **Computer Vision**: YOLO object detection for enemy/friendly target identification
- **Action Space**: Aiming (left/right movement) and discrete shooting actions
- **Reward System**: 
  - Positive rewards for eliminations
  - Negative penalties for wasteful actions
  - Strategic rewards for sustained aiming



## Results

The project demonstrated real-time interaction between the RL agent and the game environment. Training was challenging due to the complexity of the visual environment and data requirements for the world model.

## Status

Development paused for academic semester. Future work includes improving trajectory generation and world model training.
