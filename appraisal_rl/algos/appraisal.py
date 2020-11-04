import torch


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