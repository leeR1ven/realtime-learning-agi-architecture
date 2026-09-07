"""Optional extension of the original single time ring at its frontier.

Existing time identifiers never move. Only an impending wrap allocates more
positions; all modalities and all experiences still share one active clock.
This is a disclosed capacity-policy change, not an original brain mechanism.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from 海马体时间区_hippocampal_time import 时间环


class GrowingTimeRing(时间环):
    """Use the original step after ensuring it cannot wrap onto an old slot."""

    @classmethod
    def from_original(cls, original):
        activity = np.asarray(original.时间激活)
        count, current = int(original.时间神经元数量), int(original.当前时间)
        if (activity.dtype != np.bool_ or activity.shape != (count,)
                or not 0 <= current < count or int(activity.sum()) != 1
                or not activity[current]):
            raise ValueError('Cannot extend an invalid single time ring')
        result = cls(count, original.每步秒数, original.每帧步数)
        result.当前时间 = current
        result.时间激活 = activity.copy()
        return result

    def ensure_capacity_for_steps(self, steps=1):
        """Allocate before mutation, so allocation failure leaves the clock intact.

        Runtime can reserve a whole imminent frame before Hebbian updates. An
        ordinary one-step call reserves only that step. No learning table is
        changed, and no extra replay clock or per-task timeline is introduced.
        """
        if isinstance(steps, (bool, np.bool_)) or int(steps) != steps or steps < 0:
            raise ValueError('Upcoming time steps must be a nonnegative integer')
        required = int(self.当前时间) + int(steps) + 1
        old_count = int(self.时间神经元数量)
        if required <= old_count:
            return False
        largest = int(np.iinfo(np.intp).max)
        if required > largest:
            raise ValueError('Requested time capacity exceeds addressable array size')
        new_count = old_count
        while new_count < required:
            new_count = min(largest, max(new_count + 1, 2 * new_count))
        expanded = np.zeros(new_count, dtype=bool)
        expanded[:old_count] = self.时间激活
        self.时间激活 = expanded
        self.时间神经元数量 = new_count
        return True

    def 走一步(self):
        self.ensure_capacity_for_steps(1)
        return super().走一步()
