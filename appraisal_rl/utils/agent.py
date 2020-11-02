import torch
import numpy as np

from .format import *
from .storage import *
from models import ACModel, AppraisalModel


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir,
                 device=None, argmax=False, num_envs=1, use_memory=False, use_text=False, use_appraisal=False):
        obs_space, self.preprocess_obss = get_obss_preprocessor(obs_space)
        self.acmodel = ACModel(obs_space, action_space, use_memory=use_memory, use_text=use_text, use_appraisal=use_appraisal)
        self.appraisal_model = AppraisalModel(self.acmodel.embedding_size, self.acmodel.embedding_size // 2)
        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=self.device)

        self.acmodel.load_state_dict(get_model_state(model_dir))
        self.acmodel.to(self.device)
        self.acmodel.eval()

        self.appraisal = torch.zeros(3).unsqueeze(0)
        self.appraisals = [[],[],[]]
        self.appraisal_model.load_state_dict(get_appraisal_state(model_dir))
        self.appraisal_model.to(self.device)
        self.appraisal_model.eval()

        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(get_vocab(model_dir))

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories, embedding = self.acmodel(preprocessed_obss, self.memories, self.appraisal)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        appraisals = self.appraisal_model(embedding)
        self.appraisal = appraisals
        if len(self.appraisals[0]) > 20:
            for a in range(3):
                self.appraisals[a] = self.appraisals[a][1:]

        for a in range(3):
            self.appraisals[a].append(appraisals[0][a])
        
        return actions.cpu().numpy()

    def get_action(self, obs):
        action = self.get_actions([obs])
        return action[0], np.array(self.appraisals)

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=self.device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        if done:
            self.appraisals = [[],[],[]]
        return self.analyze_feedbacks([reward], [done])
