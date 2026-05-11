import math
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


def _contains_any(text: str, keywords: List[str]) -> int:
    lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in lower)


def _punctuation_intensity(text: str) -> float:
    return min(3.0, text.count("!") + text.count("?") + text.count("？") + text.count("！"))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, x))))


class AgentBase(ABC):
    """Rule-based fixed expert used to simulate a specialized agent.

    Each agent maps text + task description to a binary probability vector.
    Labels are task-local: clickbait uses 1=clickbait; sentiment uses 1=positive.
    """

    name = "Base"
    description = ""

    @abstractmethod
    def raw_score(self, text: str, task_description: str) -> Tuple[float, str]:
        """Return a class-1 logit-like score and a short explanation."""

    def predict_one(self, text: str, task_description: str) -> Dict[str, object]:
        score, explanation = self.raw_score(text, task_description)
        p1 = _sigmoid(score)
        probs = np.array([1.0 - p1, p1], dtype=np.float32)
        confidence = float(np.max(probs))
        return {"probs": probs, "confidence": confidence, "explanation": explanation}

    def predict_batch(self, texts: List[str], task_descriptions: List[str]):
        outputs = [self.predict_one(t, d) for t, d in zip(texts, task_descriptions)]
        probs = np.stack([o["probs"] for o in outputs], axis=0)
        confidences = np.array([o["confidence"] for o in outputs], dtype=np.float32)
        explanations = [o["explanation"] for o in outputs]
        return probs, confidences, explanations


class SemanticAgent(AgentBase):
    name = "Semantic"
    description = "understands semantic meaning, implication, context, and hidden meaning"

    hidden_clickbait = ["hidden", "secret", "truth", "behind", "mysterious", "真相", "秘密", "背后"]
    neutral_news = ["report", "budget", "schedule", "announces", "survey", "data", "公布", "报告", "安排"]
    positive_semantic = ["worth", "better", "warm", "kindness", "encouraged", "dependable", "安心", "感动", "明亮"]
    negative_semantic = ["regret", "wrong", "lost", "ignored", "crashed", "errors", "崩溃", "白忙", "问题"]
    negators = ["not", "no", "barely", "without", "并不", "没有", "不"]

    def raw_score(self, text: str, task_description: str) -> Tuple[float, str]:
        is_clickbait = "clickbait" in task_description.lower()
        if is_clickbait:
            pos = _contains_any(text, self.hidden_clickbait)
            neg = _contains_any(text, self.neutral_news)
            score = 0.95 * pos - 0.8 * neg - 0.15
            return score, f"semantic hidden-meaning={pos}, neutral-news={neg}"
        pos = _contains_any(text, self.positive_semantic)
        neg = _contains_any(text, self.negative_semantic)
        negation = _contains_any(text, self.negators)
        score = 0.9 * pos - 0.95 * neg - 0.15 * negation
        return score, f"semantic positive={pos}, negative={neg}, negation={negation}"


class EmotionAgent(AgentBase):
    name = "Emotion"
    description = "detects sentiment, emotion, sarcasm, subjective attitude, and implicit feeling"

    emotion_words = ["angry", "sad", "happy", "warm", "delightful", "love", "smiling", "感动", "开心", "惊喜"]
    negative_feelings = ["regret", "punishment", "ignored", "wrong", "unfortunately", "失望", "崩溃", "等待"]
    sarcasm_markers = ["amazing", "just love", "what a delightful", "of course", "apparently", "真不错", "当然开心"]
    positive_feelings = ["kindness", "smiling", "encouraged", "nicer", "respect", "安心", "明亮", "出乎意料地好"]

    def raw_score(self, text: str, task_description: str) -> Tuple[float, str]:
        is_clickbait = "clickbait" in task_description.lower()
        if is_clickbait:
            emotion = _contains_any(text, self.emotion_words)
            shock = _contains_any(text, ["shocking", "speechless", "unbelievable", "震惊", "惊人"])
            score = 0.55 * shock + 0.2 * emotion - 0.2
            return score, f"emotional sensationality={shock}, emotion={emotion}"
        pos = _contains_any(text, self.positive_feelings)
        neg = _contains_any(text, self.negative_feelings)
        sarcasm = _contains_any(text, self.sarcasm_markers)
        score = 1.05 * pos - 1.15 * neg - 1.3 * sarcasm
        return score, f"emotion positive={pos}, negative={neg}, sarcasm={sarcasm}"


class IntentionAgent(AgentBase):
    name = "Intention"
    description = "detects persuasion, misleading intention, exaggeration, and click-inducing purpose"

    click_intent = [
        "click",
        "share",
        "must see",
        "must know",
        "do not miss",
        "you need to see",
        "everyone is sharing",
        "赶紧看",
        "一定要知道",
        "点击查看",
    ]
    manipulative = ["doctors do not want", "secret", "warning sign", "could cost", "will change", "从不主动告诉"]
    calm_info = ["approves", "announces", "releases", "review", "schedule", "budget", "发布", "公布"]
    sentiment_intent = ["promised", "called it", "goal was", "apparently", "所谓", "其实"]

    def raw_score(self, text: str, task_description: str) -> Tuple[float, str]:
        is_clickbait = "clickbait" in task_description.lower()
        if is_clickbait:
            intent = _contains_any(text, self.click_intent)
            manipulation = _contains_any(text, self.manipulative)
            calm = _contains_any(text, self.calm_info)
            score = 1.15 * intent + 0.75 * manipulation - 0.9 * calm - 0.1
            return score, f"click intent={intent}, manipulation={manipulation}, calm-info={calm}"
        cue = _contains_any(text, self.sentiment_intent)
        # In sentiment, intention cues often reveal disappointment or contrast.
        positive_resolution = _contains_any(text, ["solved", "needed", "worth", "made my day", "出乎意料地好"])
        score = 0.55 * positive_resolution - 0.5 * cue
        return score, f"speaker intention/contrast={cue}, positive-resolution={positive_resolution}"


class LexicalAgent(AgentBase):
    name = "Lexical"
    description = "detects sensational words, punctuation, keywords, and surface-level patterns"

    sensational = [
        "shocking",
        "unbelievable",
        "secret",
        "weird trick",
        "must see",
        "speechless",
        "viral",
        "bizarre",
        "you won't believe",
        "震惊",
        "万万没想到",
        "竟然",
        "惊人",
    ]
    plain_news = ["report", "budget", "rules", "schedule", "survey", "summary", "measures", "报告", "预算"]
    positive_words = ["better", "kindness", "warm", "worth", "encouraged", "nicer", "respect", "安心", "感动"]
    negative_words = ["crashed", "regret", "wrong", "fee", "errors", "ignored", "punishment", "崩溃", "等待"]

    def raw_score(self, text: str, task_description: str) -> Tuple[float, str]:
        is_clickbait = "clickbait" in task_description.lower()
        punct = _punctuation_intensity(text)
        if is_clickbait:
            pos = _contains_any(text, self.sensational)
            neg = _contains_any(text, self.plain_news)
            score = 1.05 * pos + 0.2 * punct - 0.85 * neg - 0.2
            return score, f"lexical sensational={pos}, punctuation={punct:.0f}, plain={neg}"
        pos = _contains_any(text, self.positive_words)
        neg = _contains_any(text, self.negative_words)
        score = 0.85 * pos - 0.9 * neg + 0.05 * punct
        return score, f"lexical positive={pos}, negative={neg}, punctuation={punct:.0f}"


class ConsistencyAgent(AgentBase):
    name = "Consistency"
    description = "detects mismatch, contradiction, inconsistency between title, content, and image description"

    contrast = ["but", "however", "actually", "yet", "except", "unfortunately", "然而", "其实", "并不是", "虽然"]
    bait_mismatch = ["but", "actually", "hidden", "behind", "truth", "然而", "其实", "背后", "真相"]
    stable_news = ["annual", "schedule", "data", "numbers", "summary", "年度", "安排", "摘要"]
    positive_after_contrast = ["better", "worth", "warm", "needed", "good", "安心", "感动", "好"]
    negative_after_contrast = ["wrong", "errors", "crashed", "fee", "lost", "崩溃", "问题", "白忙"]

    def raw_score(self, text: str, task_description: str) -> Tuple[float, str]:
        is_clickbait = "clickbait" in task_description.lower()
        contrast = _contains_any(text, self.contrast)
        if is_clickbait:
            mismatch = _contains_any(text, self.bait_mismatch)
            stable = _contains_any(text, self.stable_news)
            score = 0.55 * contrast + 0.75 * mismatch - 0.65 * stable - 0.2
            return score, f"mismatch cues={mismatch}, contrast={contrast}, stable-news={stable}"
        pos = _contains_any(text, self.positive_after_contrast)
        neg = _contains_any(text, self.negative_after_contrast)
        # Contrast is informative, but not always negative; paired words decide direction.
        score = 0.45 * contrast + 0.75 * pos - 0.9 * neg
        return score, f"contrast={contrast}, positive-after-contrast={pos}, negative-after-contrast={neg}"


def build_agents() -> List[AgentBase]:
    return [
        SemanticAgent(),
        EmotionAgent(),
        IntentionAgent(),
        LexicalAgent(),
        ConsistencyAgent(),
    ]


def run_agents(agents: List[AgentBase], texts: List[str], task_descriptions: List[str]):
    probs, confidences, explanations = [], [], []
    for agent in agents:
        p, c, e = agent.predict_batch(texts, task_descriptions)
        probs.append(p)
        confidences.append(c)
        explanations.append(e)
    # [batch, K, classes], [batch, K]
    agent_probs = np.stack(probs, axis=1)
    agent_confidences = np.stack(confidences, axis=1)
    return agent_probs.astype(np.float32), agent_confidences.astype(np.float32), explanations
