"""UCB1 over mutation families.

The search loop pulls a mutation "arm" (identified by its family name), sees
whether the mutation improved the fitness metric, and feeds that back. UCB1
then decides which families to try first in later rounds: unpulled arms are
always tried, and arms with a better track record are preferred, with an
exploration bonus that decays as an arm is pulled more.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class _Arm:
    pulls: int = 0
    reward_sum: float = 0.0

    @property
    def mean(self) -> float:
        return self.reward_sum / self.pulls if self.pulls else 0.0


@dataclass
class UCB1:
    c: float = 1.4
    arms: dict[str, _Arm] = field(default_factory=dict)
    total_pulls: int = 0

    def _arm(self, name: str) -> _Arm:
        return self.arms.setdefault(name, _Arm())

    def update(self, name: str, reward: float) -> None:
        arm = self._arm(name)
        arm.pulls += 1
        arm.reward_sum += reward
        self.total_pulls += 1

    def score(self, name: str) -> float:
        arm = self._arm(name)
        if arm.pulls == 0:
            return math.inf
        bonus = self.c * math.sqrt(math.log(max(self.total_pulls, 1)) / arm.pulls)
        return arm.mean + bonus

    def order(self, names) -> list[str]:
        """Return the names sorted best-first by current UCB score."""
        return sorted(names, key=lambda n: self.score(n), reverse=True)

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            n: {"pulls": a.pulls, "mean_reward": round(a.mean, 4)}
            for n, a in sorted(self.arms.items())
        }
