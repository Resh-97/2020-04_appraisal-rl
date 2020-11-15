# appraisal-rl
Emotional appraisal in Reinforcement Learning Agents.

We implement a cognitive form of emotion in reinforcement learning agents, who perform appraisals of their situation and alter their behavior based on the emotion elicited. In particular, we formulate three appraisal variables: motivational relevance, novelty, and accountability, that reinforcement learning agents can use to appraise fully or partially-observable state representations from the environment. We design a neural network based agent architecture and propose environments and learning algorithms to learn internal models of these appraisal variables over interactions between each agent and the environment.

## Usage
1. Install the included `gym_minigrid` as a package
```
pip install -e gym-minigrid
```
2. Run the training, visualization, and evaluation scripts. To obtain the 
```
cd appraisal_rl
python train.py/visualize.py/evaluate.py --flags
```
3. To obtain the result in our report, train:
```
python train.py --algo ppo --env MiniGrid-Dynamic-Obstacles-Random-10x10-v0 --model 10x10_Appraisal --frames 500000 --recurrence 4 --lr 5e-3 --batch-size 360   
```
4. Visualize the result:
```
python visualize.py --env MiniGrid-Dynamic-Obstacles-Random-10x10-v0 --model 10x10_Appraisal
```
5. You can use `tensorboard` to view the training logs.
```
tensorboard --logdir storage
```

## Credits
- gym-minigrid: https://github.com/maximecb/gym-minigrid
- rl-starter-files: https://github.com/lcswillems/rl-starter-files
- torch-ac: https://github.com/lcswillems/torch-ac
