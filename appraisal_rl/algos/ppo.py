import numpy
import torch
import torch.nn.functional as F

from .base import BaseAlgo

def motivational_relevance(obs):
    """Computes motivational relevance for a batch of observations.
    Motivational relevance is a function of the L1 distance to the goal.
    Some observation states do not contain the goal, so relevance is zero.
    """
    batch_size, w, _ = obs.size()
    relevance = torch.zeros(batch_size)
    agent_pos = torch.nonzero(obs == 10)[:, 1:]
    goal_poss = torch.nonzero(obs == 8)
    for goal in goal_poss:
        idx, goal_pos = goal[0], goal[1:]
        dist = torch.norm(agent_pos[idx] - goal_pos.float(), 1)
        relevance[idx] = 1 - (dist - 1) / (2 * (w - 1))
    return relevance

def novelty(logits):
    """Computes novelty according to the KL Divergence from perfect uncertainty.
    The higher the KL Divergence, the less novel the scenario,
    so we take novelty as the negative of the KL Divergence.
    """
    batch_size, num_actions = logits.size()
    P = torch.softmax(logits, dim=1)
    Q = torch.full(P.size(), 1 / num_actions)
    return -torch.sum(Q * torch.log(Q / P), dim=1)

def accountability(actions, reward):
    """How responsible were you for the outcome that happened?
    If a highly probable observation was observed after an action, it can
    be assumed that the agent's action caused the state change, and vice-versa.
    Accountability is in the range 0 (not caused by agent) to 1 (caused by agent).

    Regret
    The overall feeling of regret at some decision is a combination of these two components: 
        (1) You regret both that the outcome is poorer than some standard (often the outcome of the option you rejected) and,
        (2) that the decision you made was, in retrospect, unjustified.
    """ 
    return reward


class PPOAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, appraisal_model, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, appraisal_model, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.appraisal_optimizer = torch.optim.Adam(self.appraisal_model.parameters(), lr, eps=adam_eps)
        self.batch_num = 0

    def update_parameters(self, exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []
            log_appraisal_losses = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0
                batch_appraisal_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]
                
                appraisals = None

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]
                    if appraisals is None:
                        appraisals = torch.zeros((len(inds), 3))

                    # Compute loss

                    if self.acmodel.recurrent:
                        dist, value, memory, embedding = self.acmodel(sb.obs, memory * sb.mask, appraisals)
                    else:
                        dist, value = self.acmodel(sb.obs)

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
                    
                    appraisals = self.appraisal_model(embedding)
                    appraisal_target = torch.vstack((
                        motivational_relevance(sb.obs.image[..., 0]),
                        novelty(dist.logits),
                        accountability(sb.action, sb.reward)
                    )).T
                    appraisal_loss = F.mse_loss(appraisals, appraisal_target)

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss
                    batch_appraisal_loss += appraisal_loss.item()

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence
                batch_appraisal_loss /= self.recurrence

                # Update appraisal

                self.appraisal_optimizer.zero_grad()
                appraisal_loss.backward(retain_graph=True)
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.appraisal_model.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.appraisal_model.parameters(), self.max_grad_norm)
                self.appraisal_optimizer.step()

                # Update actor-critic,

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)
                log_appraisal_losses.append(batch_appraisal_loss)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms),
            "appraisal_loss": numpy.mean(log_appraisal_losses)
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
