# MuJoCo_ARH_Def_Grasp



### Deformable Grasping Simulation using MuJoCo (v3.3.7)



This repository contains simulation environments and scripts for studying **deformable object grasping** using both **parallel-jaw grippers** and **Anthropomorphic Robot Hands (ARHs)** in [MuJoCo](https://github.com/google-deepmind/mujoco).  

The project aims to build modular, physics-based setups for benchmarking grasp performance, contact behavior, and control strategies across different gripper configurations.



---



## Overview



The project is organized into two phases:



1. **Phase 1 — Parallel-Jaw Baseline**
   - Develops and validates grasp simulations using a simple parallel-jaw gripper.
   - Focuses on simulating deformable object contact, compliance response, and grasp stability.
   - Serves as the foundation for control pipelines and physics tuning.



2. **Phase 2 — Anthropomorphic Robot Hand (ARH) Integration**
   - Introduces a multi-DOF anthropomorphic hand for complex, human-like grasping.
   - Expands simulation capabilities for dexterous and adaptive manipulation tasks.
   - Targets teleoperation and reinforcement learning pipelines in later stages.



---



## Features



- MuJoCo-based physics simulation with deformable body support.  

- Configurable object stiffness and shape for comparative grasp studies.  

- Extensible architecture for robot hand integration and controller design.  

- Reference XMLs and assets inspired by official MuJoCo examples.  

- Clean modular Python scripts for loading, controlling, and analyzing simulations.  



---


## Project Structure


```
mujoco3_ARH_def_grasp/
|
+-- assets/                       # Object meshes for simulation
|   
+-- backup/                       # Saved prior working scripts and models
|
+-- data/                         # Logged contacts and outputs
|   \-- cube_contact.csv
|
+-- models/                       # XML model files and scene definitions
|   \-- parallel-jaw_gripper.xml  # Parallel-jaw gripper + soft cube
|
+-- scripts/                      # Python scripts for running simulations
|   \-- parallel-jaw_grasp_sim.py
|
+-- external/                     # MuJoCo example assets (subset)
|
+-- .gitignore                    # Python cache and temporary files
+-- requirements.txt              # Project dependencies (MuJoCo v3.3.7)
+-- README.md                     # Project documentation (this file)
```



