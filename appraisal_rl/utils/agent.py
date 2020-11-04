import torch
import numpy as np

from .format import *
from .storage import *
from models import ACModel


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir,
                 device=None, argmax=False, num_envs=1, use_text=False):
        obs_space, self.preprocess_obss = get_obss_preprocessor(obs_space)
        self.acmodel = ACModel(obs_space, action_space, use_text=use_text)
        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs

        self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=self.device)
        self.appraisal = [list() for _ in range(3)]

        self.acmodel.load_state_dict(get_model_state(model_dir))
        self.acmodel.to(self.device)
        self.acmodel.eval()

        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(get_vocab(model_dir))

    def get_actions(self, obss, dist, appraisal):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        with torch.no_grad():
            dist, _, self.memories, embedding, appraisal = self.acmodel(preprocessed_obss, self.memories, dist, appraisal)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        for i, app in enumerate(appraisal[0]):
            self.appraisal[i].append(app.item())

        if len(self.appraisal[0]) > 20:
            for i, app in enumerate(self.appraisal):
                self.appraisal[i] = app[1:]
        
        return dist, actions.cpu().numpy()

    def get_action(self, obs, dist, appraisal):
        dist, action = self.get_actions([obs], dist, appraisal)
        return dist, action[0], self.appraisal

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=self.device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        if done:
            self.appraisals = [[],[],[]]
        return self.analyze_feedbacks([reward], [done])
