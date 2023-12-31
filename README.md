# Implementing vanilla DQN model 
A simple implementation in Pytorch of vanilla DQN model, featured in the paper Playing Atari with Deep Reinforcement Learning (2013) [arXiv:1312.5602](https://doi.org/10.48550/arXiv.1312.5602)

### Environment specifications:     
Python version:  3.9.15   
Default replay memory might takes up a lot of memory, ~20 GBs    
Dependencies:     
| Package   |  Version | Installation note|
|-----------|----------|------------------|
| gym       |  0.21.0  |        N/A        |  
|torch      | 2.1.0.dev20230526 |refer to [official installation page](https://pytorch.org/get-started/locally/) |
|tensorboard|2.11.0    | N/A |
|matplotlib |3.6.2     | N/A |

** No installation note means package can simply be installed via pip

------ 

## Demo 

### Pong
Model after 2,500,000 training steps (Learning rate=0.00025)   

[Pong Demo Vid](https://github.com/quingzz/DQN_paper/assets/90673616/fc9efb13-9dbf-4354-8dfc-808aa2b8c226)

### Breakout 
Model after 24,000,000 training steps (Learning rate=0.000025, wihout penalizing lose lives)    
***Note:*** The model prone to perform random actions instead of FIRE (to reset game) after losing lives (which was cut out by force reset)


[Breakout No Penalize Demo Vid](https://github.com/quingzz/DQN_paper/assets/90673616/f14bde91-1c6e-41d8-90e1-4173263aed72)

Model (Learning rate=0.000025, with penalizing lose lives) (to be updated)    
N/A
