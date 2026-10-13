"""ms-swift reward plugin for A2Seek GRPO training."""

from pathlib import Path
import sys

# ms-swift imports external plugins by filename in a separate process.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swift.plugin import ORM, orms
from a2seek.reward_core import box_score, class_score, format_score, length_score, seeking_score


def batch_score(completions, solution, scorer):
    if len(completions) != len(solution):
        raise ValueError("Each GRPO completion must have a corresponding solution")
    return [scorer(completion, reference) for completion, reference in zip(completions, solution)]


class ThoughtChainFormat(ORM):
    def __call__(self, completions, **kwargs):
        return [format_score(completion) for completion in completions]


class CLS_Accuracy(ORM):
    def __call__(self, completions, solution, **kwargs):
        return batch_score(completions, solution, class_score)


class BBox_Accuracy(ORM):
    def __call__(self, completions, solution, **kwargs):
        return batch_score(completions, solution, box_score)


class SeekingAccuracy(ORM):
    def __call__(self, completions, solution, **kwargs):
        return batch_score(completions, solution, seeking_score)


class DynamicThinkingLength(ORM):
    # ms-swift passes both the training tokenizer and this GRPO argument.
    def __init__(self, tokenizer, max_completion_length=512):
        if tokenizer is None or max_completion_length <= 0:
            raise ValueError("Length reward requires a tokenizer and positive max_completion_length")
        self.tokenizer = tokenizer
        self.max_completion_length = max_completion_length

    def __call__(self, completions, solution, **kwargs):
        return batch_score(completions, solution,
                           lambda c, s: length_score(c, s, self.tokenizer, self.max_completion_length))


orms.update({"format_reward": ThoughtChainFormat, "cls_accuracy": CLS_Accuracy,
             "bbox_accuracy": BBox_Accuracy, "length_reward": DynamicThinkingLength,
             "seeking_reward": SeekingAccuracy})
