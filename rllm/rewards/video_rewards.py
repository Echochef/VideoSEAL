"""Video reward entrypoints.

Kept as a thin wrapper so callers can import `reward_video` without depending on
the internal file layout.
"""

from rllm.rewards.video_reward_dp import reward_video  # noqa: F401

