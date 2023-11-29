from typing import Callable, Optional, Sequence, Tuple, List
import torch
from torch import nn


from cs285.agents.dqn_agent import DQNAgent


class AWACAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        temperature: float,
        **kwargs,
    ):
        super().__init__(observation_shape=observation_shape, num_actions=num_actions, **kwargs)

        self.actor = make_actor(observation_shape, num_actions)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.temperature = temperature

    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        with torch.no_grad():
            # TODO(student): compute the actor distribution, then use it to compute E[Q(s, a)]
            next_actions_probs = self.actor(next_observations)
            next_actions = next_action_prob.sample()

            # Compute expected Q values for next states
            next_qa_values = self.critic(next_observations, next_actions).detach()
            next_qs = torch.sum(next_actions_probs * next_qa_values, dim=1)

            # Compute the TD target
            target_values = rewards + self.gamma * next_qs * (1 - dones)



        # TODO(student): Compute Q(s, a) and loss similar to DQN
        q_values = self.critic(observations, actions)
        assert q_values.shape == target_values.shape

        loss = torch.nn.functional.mse_loss(q_values, target_values)

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): compute the advantage of the actions compared to E[Q(s, a)]
        action_probs = action_dist.probs if action_dist is not None else self.actor(observations)
        qa_values = self.critic(observations, actions)
        q_values = torch.sum(action_probs * self.critic(observations), dim=1)

        advantages = qa_values - q_values.unsqueeze(-1)
        return advantages

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        # TODO(student): update the actor using AWAC
        # Compute the current policy's action distribution
        current_action_dist = self.actor(observations)
        log_probs = current_action_dist.log_prob(actions)

        # Compute advantages
        advantages = self.compute_advantage(observations, actions, current_action_dist)

        # Calculate the actor loss
        loss = -(log_probs * torch.exp(advantages / self.temperature)).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()

    def update(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, dones: torch.Tensor, step: int):
        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the actor.
        actor_loss = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_loss

        return metrics
