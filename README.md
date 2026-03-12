PyFAC
=
# Introduction

_Reinforcement Learning of Robust Active Flow Control Strategy with Surrogate-Based Hybrid Environment_

**Abstract:**
This article introduces a training framework for Active Flow Control (AFC) based on Deep Reinforcement Learning (DRL) called PyFAC, 
which employs the Soft Actor-Critic (SAC) algorithm as the DRL method. 
To reduce computational resource consumption and improve training efficiency, 
a Long Short-Term Memory (LSTM) model is incorporated as a surrogate environment for Computational Fluid Dynamics (CFD). 
Additionally, a Free-Stream Turbulence (FST) model is introduced to validate the robustness of the trained AFC agent under disturbing incoming flow conditions. 
Under the condition without incoming flow disturbance, the agent gains 3.16% extra lift compared with baseline which is not stimulated by AFC, 
and 28.3% improvement of control performance compared with ordinary control strategies without the involvement of DRL. 
It is observed that an agent trained in a plain condition without FST struggles to perform effectively in an FST environment due to the disturbances from the incoming flow. 
Conversely, introducing FST perturbations into the training environment allows for the development of a robust agent. Subsequently, 
the LSTM model is integrated as the CFD surrogate. The LSTM demonstrates high accuracy, with discrepancies compared to CFD remaining below 10-3. 
Agents trained in the LSTM surrogate environment producing control strategies comparable to those of the model-free agents and achieving nearly identical control performances. 
A comparative analysis of training times for LSTM-based and model-free DRL approaches is also conducted, revealing a 60% improvement in training efficiency with the LSTM-based method. 

![image](https://github.com/noahhanfzh/PyFAC/blob/main/PyFAC_Structure.png)

# Requirement
python==3.11

ansys-fluent-core==0.35.0

gymnasium==1.2.2

matplotlib==3.10.7

numpy==2.3.4

torch==2.7.1

stable_baselines3==2.7.0

ANSYS==2024 R1

# Code Guide
**program/Main.py:** main program

**program/Input_Setup.py:** user input

**program/PyFAC_2D_3D.py:** training setup

**program/Callback.py:** Callback

**program/Define_Model.py:** init DRL model

**program/Plot_Output.py:** plot output

**program/PyFAC_Env.py:** training environment

**program/LSTMenv.py:** LSTM environment

**pretrained LSTM model:** pretrained LSTM model

**pretrained agent model:** pretrained agent model
