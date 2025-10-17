import math
from base_model import BaseDeepQModel
import numpy as np

#  Epsilon controller (panic bump). basically the modifitcaiton from the baseline, so that my strategy works
class EpsilonController:
    def __init__(self, base_schedule="exp",
                 eps_start=0.5, eps_min=0.02, max_cap=0.5,
                 decay_steps=60_000, tau=30_000,
                 bad_bar=120, bump=0.08,
                 decay_back=0.97, cooldown_eps=5):
        self.base = base_schedule
        self.start, self.min, self.max_cap = eps_start, eps_min, max_cap
        self.decay_steps, self.tau = decay_steps, tau
        self.bad_bar, self.bump = bad_bar, bump
        self.decay_back, self.cooldown_eps = decay_back, cooldown_eps
        self.step = 0
        self.eps = self.start
        self._cooldown = 0

    def _base_eps(self):
        if self.base == "exp":
            return self.min + (self.start - self.min) * math.exp(-self.step / self.tau)
        frac = min(1.0, self.step / self.decay_steps)
        return max(self.min, self.start - (self.start - self.min) * frac)

    def on_step(self):
        self.step += 1
        self.eps = max(self.min, min(self.max_cap, self._base_eps()))
        return self.eps

    def on_episode_end(self, total_reward):
        self.eps = max(self.min, min(self.max_cap, self.eps * self.decay_back))
        if self._cooldown > 0:
            self._cooldown -= 1
            return self.eps
        if total_reward < self.bad_bar:
            self.eps = min(self.max_cap, self.eps + self.bump)
            self._cooldown = self.cooldown_eps
        return self.eps

class PanicBumpModel(BaseDeepQModel):
    def train_loop(self,
                episodes_per_length=300,
                t_net_update_freq=50, sample_batch=64,
                warmup_steps=1000,
                pole_lengths=None,     # <--- NEW
                **kwargs):

        eps_ctl = EpsilonController(
            base_schedule="linear",
            eps_min=0.05, eps_max=1.0,
            decay_steps=200_000, tau=100_000,
            bad_bar=200, bump=0.20, decay_back=0.98, cooldown_eps=3
        )

        episode_rewards = []
        total_steps = 0
        warmup = True

        for episode in range(episodes_per_length):
            if warmup and total_steps > warmup_steps:
                warmup = False

            state, _ = self.env.reset()
            done = False
            ep_reward = 0

            while not done:
                eps = 1.0 if warmup else eps_ctl.on_step()
                action = self.choose_action(state, epsilon=eps, warmup=warmup)
                next_state, reward, done, _, _ = self.env.step(action)

                self.memory_buffer.push((state, action, reward, next_state, done))
                state = next_state
                ep_reward += reward
                total_steps += 1

                if warmup:
                    continue

                loss = self.calculate_loss(sample_batch)
                if loss is not None:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if not warmup:
                eps_ctl.on_episode_end(ep_reward)

            episode_rewards.append(ep_reward)
            avg50 = sum(episode_rewards[-50:]) / min(50, len(episode_rewards))
            print(f"\r Ep {episode+1:4d}/{episodes_per_length} | "
                f"R={ep_reward:4.0f} | eps={eps_ctl.eps: .3f} | avg50={avg50:6.1f}",
                end="", flush=True)

            if (episode + 1) % t_net_update_freq == 0:
                self.update_target_network()




if __name__ == "__main__":
    pole_lengths = np.linspace(0.4, 1.8, 3)
    model = PanicBumpModel(state_dim=4, action_dim=2, memory_buffer_size=50000, learning_rate=0.0001, discount_factor=0.5)
    print('Initial Training')
    print(' ----------------------- ')
    model.train_loop( episodes_per_length=500, warmup_steps=1000, sample_batch=256, warmup=True)
    for length in pole_lengths:
        print('Training for pole length:', length)
        print(' ----------------------- ')
        model.update_pole_length(length)
        model.train_loop(500, epsilon=0.2, warmup_steps=0, sample_batch=256, warmup=False)


    model.save_model('panic_bump_eps')