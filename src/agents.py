import json
import math
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from .task_schemas import DEFAULT_LABEL_SCHEMA, get_task_label_schema


def _label_schema_for_sample(
    explicit_schema: Optional[Dict[str, str]],
    task_description: str,
    record: Optional[Dict[str, object]] = None,
) -> Dict[str, str]:
    if explicit_schema:
        return explicit_schema
    task_name = record.get("task_name") if record else None
    return get_task_label_schema(task_name, task_description)

def _contains_any(text: str, keywords: List[str]) -> int:
    lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in lower)


def _punctuation_intensity(text: str) -> float:
    return min(3.0, text.count("!") + text.count("?") + text.count("！") + text.count("？"))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, x))))


def _normalize_probs(p0: float, p1: float) -> Tuple[float, float]:
    p0 = float(p0)
    p1 = float(p1)
    if p0 < 0.0 or p1 < 0.0:
        p0 = min(1.0, max(0.0, p0))
        p1 = min(1.0, max(0.0, p1))
    total = p0 + p1
    if total <= 1e-8:
        return 0.5, 0.5
    return min(1.0, max(0.0, p0 / total)), min(1.0, max(0.0, p1 / total))


def _clamp01(value: float, default: float = 0.5) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _first_json_object(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("No JSON object found in LLM output")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(cleaned)):
        char = cleaned[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : idx + 1]
    raise ValueError("Unclosed JSON object in LLM output")


class AgentInterface(ABC):
    name = "Base"
    description = ""

    @abstractmethod
    def predict_one(
        self,
        text: str,
        task_description: str,
        label_schema: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        """Return probs/confidence/explanation for one sample."""

    def predict_batch(
        self,
        texts: List[str],
        task_descriptions: List[str],
        records: Optional[List[Dict[str, object]]] = None,
        label_schema: Optional[Dict[str, str]] = None,
    ):
        records = records or [None] * len(texts)
        outputs = [
            self.predict_one(t, d, _label_schema_for_sample(label_schema, d, record))
            for t, d, record in zip(texts, task_descriptions, records)
        ]
        probs = np.stack([np.asarray(o["probs"], dtype=np.float32) for o in outputs], axis=0)
        confidences = np.array([o["confidence"] for o in outputs], dtype=np.float32)
        explanations = [str(o["explanation"]) for o in outputs]
        return probs, confidences, explanations


class RuleBasedAgent(AgentInterface):
    """Rule-based fixed expert used to simulate a specialized agent."""

    @abstractmethod
    def raw_score(self, text: str, task_description: str) -> Tuple[float, str]:
        """Return a class-1 logit-like score and a short explanation."""

    def predict_one(
        self,
        text: str,
        task_description: str,
        label_schema: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        score, explanation = self.raw_score(text, task_description)
        p1 = _sigmoid(score)
        probs = np.array([1.0 - p1, p1], dtype=np.float32)
        confidence = float(np.max(probs))
        return {"probs": probs, "confidence": confidence, "explanation": explanation}


class SemanticAgent(RuleBasedAgent):
    name = "Semantic"
    description = "understands semantic meaning, implication, context, and hidden meaning"

    hidden_clickbait = ["hidden", "secret", "truth", "behind", "mysterious", "revealed", "unknown", "真相", "秘密", "背后"]
    neutral_news = ["report", "budget", "schedule", "announces", "survey", "data", "review", "summary", "报告", "公布"]
    positive_semantic = ["worth", "better", "warm", "kindness", "encouraged", "dependable", "helpful", "安心", "感动"]
    negative_semantic = ["regret", "wrong", "lost", "ignored", "crashed", "errors", "failed", "崩溃", "白忙"]
    negators = ["not", "no", "barely", "without", "never", "不", "没有", "并不"]

    def raw_score(self, text: str, task_description: str) -> Tuple[float, str]:
        is_clickbait = "clickbait" in task_description.lower() or "点击" in task_description
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


class EmotionAgent(RuleBasedAgent):
    name = "Emotion"
    description = "detects sentiment, emotion, sarcasm, subjective attitude, and implicit feeling"

    emotion_words = ["angry", "sad", "happy", "warm", "delightful", "love", "smiling", "开心", "震惊", "感动"]
    negative_feelings = ["regret", "punishment", "ignored", "wrong", "unfortunately", "sad", "angry", "失望", "崩溃"]
    sarcasm_markers = ["amazing", "just love", "what a delightful", "of course", "apparently", "当然开心", "真不错"]
    positive_feelings = ["kindness", "smiling", "encouraged", "nicer", "respect", "warm", "happy", "安心", "感动"]

    def raw_score(self, text: str, task_description: str) -> Tuple[float, str]:
        is_clickbait = "clickbait" in task_description.lower() or "点击" in task_description
        if is_clickbait:
            emotion = _contains_any(text, self.emotion_words)
            shock = _contains_any(text, ["shocking", "speechless", "unbelievable", "amazing", "震惊", "惊人"])
            score = 0.55 * shock + 0.2 * emotion - 0.2
            return score, f"emotional sensationality={shock}, emotion={emotion}"
        pos = _contains_any(text, self.positive_feelings)
        neg = _contains_any(text, self.negative_feelings)
        sarcasm = _contains_any(text, self.sarcasm_markers)
        score = 1.05 * pos - 1.15 * neg - 1.3 * sarcasm
        return score, f"emotion positive={pos}, negative={neg}, sarcasm={sarcasm}"


class IntentionAgent(RuleBasedAgent):
    name = "Intention"
    description = "detects persuasion, misleading intention, exaggeration, and click-inducing purpose"

    click_intent = ["click", "share", "must see", "must know", "do not miss", "you need to see", "everyone is sharing", "点击", "速看", "必看", "一定要知道"]
    manipulative = ["doctors do not want", "secret", "warning sign", "could cost", "will change", "不告诉你", "真相"]
    calm_info = ["approves", "announces", "releases", "review", "schedule", "budget", "report", "发布", "报告"]
    sentiment_intent = ["promised", "called it", "goal was", "apparently", "其实", "所谓"]

    def raw_score(self, text: str, task_description: str) -> Tuple[float, str]:
        is_clickbait = "clickbait" in task_description.lower() or "点击" in task_description
        if is_clickbait:
            intent = _contains_any(text, self.click_intent)
            manipulation = _contains_any(text, self.manipulative)
            calm = _contains_any(text, self.calm_info)
            score = 1.15 * intent + 0.75 * manipulation - 0.9 * calm - 0.1
            return score, f"click intent={intent}, manipulation={manipulation}, calm-info={calm}"
        cue = _contains_any(text, self.sentiment_intent)
        positive_resolution = _contains_any(text, ["solved", "needed", "worth", "made my day", "helped", "出乎意料地好"])
        score = 0.55 * positive_resolution - 0.5 * cue
        return score, f"speaker intention/contrast={cue}, positive-resolution={positive_resolution}"


class LexicalAgent(RuleBasedAgent):
    name = "Lexical"
    description = "detects sensational words, punctuation, keywords, and surface-level patterns"

    sensational = ["shocking", "unbelievable", "secret", "weird trick", "must see", "speechless", "viral", "bizarre", "you won't believe", "震惊", "竟然", "万万没想到", "惊人"]
    plain_news = ["report", "budget", "rules", "schedule", "survey", "summary", "measures", "报告", "预算"]
    positive_words = ["better", "kindness", "warm", "worth", "encouraged", "nicer", "respect", "安心", "感动"]
    negative_words = ["crashed", "regret", "wrong", "fee", "errors", "ignored", "punishment", "崩溃", "等待"]

    def raw_score(self, text: str, task_description: str) -> Tuple[float, str]:
        is_clickbait = "clickbait" in task_description.lower() or "点击" in task_description
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


class ConsistencyAgent(RuleBasedAgent):
    name = "Consistency"
    description = "detects mismatch, contradiction, inconsistency between title, content, and image description"

    contrast = ["but", "however", "actually", "yet", "except", "unfortunately", "然而", "其实", "但是"]
    bait_mismatch = ["but", "actually", "hidden", "behind", "truth", "然而", "其实", "背后", "真相"]
    stable_news = ["annual", "schedule", "data", "numbers", "summary", "report", "年度", "摘要"]
    positive_after_contrast = ["better", "worth", "warm", "needed", "good", "helpful", "安心", "感动", "好"]
    negative_after_contrast = ["wrong", "errors", "crashed", "fee", "lost", "failed", "崩溃", "白忙"]

    def raw_score(self, text: str, task_description: str) -> Tuple[float, str]:
        is_clickbait = "clickbait" in task_description.lower() or "点击" in task_description
        contrast = _contains_any(text, self.contrast)
        if is_clickbait:
            mismatch = _contains_any(text, self.bait_mismatch)
            stable = _contains_any(text, self.stable_news)
            score = 0.55 * contrast + 0.75 * mismatch - 0.65 * stable - 0.2
            return score, f"mismatch cues={mismatch}, contrast={contrast}, stable-news={stable}"
        pos = _contains_any(text, self.positive_after_contrast)
        neg = _contains_any(text, self.negative_after_contrast)
        score = 0.45 * contrast + 0.75 * pos - 0.9 * neg
        return score, f"contrast={contrast}, positive-after-contrast={pos}, negative-after-contrast={neg}"


class LLMAgentError(RuntimeError):
    pass


class LLMBaseAgent(AgentInterface):
    """LLM-backed expert. It is inference-only and never participates in backprop."""

    system_role = "通用专家"
    perspective = "从自己的专家视角判断文本是否属于目标类别。"
    input_mode = "title_content"
    fewshot_file = None

    def __init__(self, llm_client, cache=None, fallback_agent: Optional[RuleBasedAgent] = None):
        self.llm_client = llm_client
        self.cache = cache
        self.fallback_agent = fallback_agent
        self.cache_only = False
        self.last_error = None
        self.last_used_fallback = False
        self.fewshot_examples = self._load_fewshot_examples()
    def set_cache_only(self, cache_only: bool = True) -> None:
        self.cache_only = cache_only

    def _load_fewshot_examples(self) -> Optional[List[Dict]]:
        """Load few-shot examples from JSON file if specified."""
        if not self.fewshot_file:
            return None
        
        try:
            from config import FEWSHOT_ENABLED, FEWSHOT_DIR
            if not FEWSHOT_ENABLED:
                return None
            
            filepath = FEWSHOT_DIR / self.fewshot_file
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    examples = json.load(f)
                return examples
        except Exception as e:
            print(f"Warning: Failed to load few-shot examples for {self.name}: {e}")
        
        return None

    @property
    def model_name(self) -> str:
        return getattr(self.llm_client, "model", "unknown")

    def system_prompt(self) -> str:
        return (
            f"你是{self.system_role}。{self.perspective}\n"
            "你只从自己的专家视角判断，不要综合其他角度。\n"
            "这是一个二分类文本判断任务，具体任务由用户提示中的 task_description 和 label_schema 决定。\n"
            "class_0 和 class_1 的具体含义以用户提示中的标签定义为准。\n"
            "probability 必须在 0 到 1 之间；confidence 表示你对自己判断的可靠程度。\n"
            "输出必须是严格 JSON，不要 Markdown，不要代码块，不要额外解释。"
        )

    def user_prompt(self, text: str, task_description: str, label_schema: Dict[str, str]) -> str:
        label_schema = label_schema or DEFAULT_LABEL_SCHEMA
        prompt_parts = []
        
        prompt_parts.append("任务描述：\n")
        prompt_parts.append(f"{task_description}\n\n")
        
        prompt_parts.append("标签定义：\n")
        prompt_parts.append(f"class_0 = {label_schema.get('class_0', DEFAULT_LABEL_SCHEMA['class_0'])}\n")
        prompt_parts.append(f"class_1 = {label_schema.get('class_1', DEFAULT_LABEL_SCHEMA['class_1'])}\n\n")
        
        if self.fewshot_examples:
            prompt_parts.append("以下是一些示例，供你参考判断标准：\n\n")
            for idx, example in enumerate(self.fewshot_examples, 1):
                prompt_parts.append(f"示例 {idx}：\n")
                prompt_parts.append(f"文本：{example['example']}\n")
                prompt_parts.append(f"输出：\n")
                prompt_parts.append(json.dumps({
                    "class_0_probability": example["class_0_probability"],
                    "class_1_probability": example["class_1_probability"],
                    "confidence": example["confidence"],
                    "explanation": example["explanation"]
                }, ensure_ascii=False, indent=2))
                prompt_parts.append("\n\n")
            
            prompt_parts.append("请按照上述示例的判断标准，对待判断文本进行分析。\n\n")
        
        prompt_parts.append("待判断文本：\n")
        prompt_parts.append(f"{text}\n\n")
        prompt_parts.append("请从你的专家视角输出严格 JSON：\n")
        prompt_parts.append("{\n")
        prompt_parts.append('  "class_0_probability": 0.0,\n')
        prompt_parts.append('  "class_1_probability": 1.0,\n')
        prompt_parts.append('  "confidence": 0.0,\n')
        prompt_parts.append('  "explanation": "不超过50字的中文解释"\n')
        prompt_parts.append("}")
        
        return "".join(prompt_parts)

    @staticmethod
    def _clean_source_value(value: object) -> str:
        if value is None:
            return ""
        try:
            if isinstance(value, float) and math.isnan(value):
                return ""
        except TypeError:
            pass
        return str(value).strip()

    def generate_input_text(
        self,
        text: str,
        record: Optional[Dict[str, object]] = None,
    ) -> str:
        """Build the LLM-visible input while leaving rule agents on df["text"].

        LLM input policy:
        - Lexical, Intention, Emotion: title only.
        - Semantic, Consistency: title + content.
        """
        fallback_text = self._clean_source_value(text)
        if not record:
            return fallback_text

        title = self._clean_source_value(record.get("title"))
        content = self._clean_source_value(record.get("content"))
        if self.input_mode == "title":
            return title or fallback_text
        if title and content:
            return f"{title}\n{content}".strip()
        return title or content or fallback_text

    @staticmethod
    def _parse_llm_json(raw_text: str) -> Dict[str, object]:
        try:
            cleaned = _first_json_object(raw_text)
            data = json.loads(cleaned)
        except (ValueError, json.JSONDecodeError) as exc:
            raise LLMAgentError(f"LLM JSON parse failed: {exc}") from exc

        try:
            p0, p1 = _normalize_probs(
                data["class_0_probability"],
                data["class_1_probability"],
            )
        except KeyError as exc:
            raise LLMAgentError(f"LLM JSON missing field: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise LLMAgentError(f"LLM probability field is invalid: {exc}") from exc
        confidence = _clamp01(data.get("confidence", max(p0, p1)), default=max(p0, p1))
        explanation = str(data.get("explanation", "LLM explanation missing")).strip()
        if not explanation:
            explanation = "LLM explanation missing"
        return {
            "probs": np.array([p0, p1], dtype=np.float32),
            "confidence": confidence,
            "explanation": explanation[:120],
        }

    @staticmethod
    def neutral_fallback() -> Dict[str, object]:
        raise LLMAgentError("Neutral LLM fallback is disabled because it contaminates training data.")

    def _cache_output(
        self,
        text: str,
        task_description: str,
        output: Dict[str, object],
        raw_text: str,
    ) -> None:
        if self.cache is not None:
            self.cache.set(text, task_description, self.name, self.model_name, output, raw_text)

    def _cache_failure(
        self,
        text: str,
        task_description: str,
        error_message: object,
    ) -> None:
        if self.cache is not None:
            self.cache.set_failed(text, task_description, self.name, self.model_name, error_message)

    def _predict_one_prepared(
        self,
        llm_text: str,
        fallback_text: str,
        task_description: str,
        label_schema: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        self.last_error = None
        self.last_used_fallback = False
        label_schema = label_schema or get_task_label_schema(task_description=task_description)
        if self.cache is not None:
            cached = self.cache.get(llm_text, task_description, self.name, self.model_name)
            if cached is not None:
                return cached

        if self.cache_only:
            self.last_error = "LLM cache miss during cache-only training. Run scripts/precompute_llm_outputs.py first."
            self.last_used_fallback = True
            raise LLMAgentError(self.last_error)

        try:
            raw_text = self.llm_client.complete(
                messages=[
                    {"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": self.user_prompt(llm_text, task_description, label_schema)},
                ]
            )
            parsed = self._parse_llm_json(raw_text)
            self._cache_output(llm_text, task_description, parsed, raw_text)
            return parsed
        except Exception as exc:
            self.last_error = str(exc)
            self.last_used_fallback = True
            self._cache_failure(llm_text, task_description, self.last_error)
            if self.fallback_agent is not None:
                fallback = self.fallback_agent.predict_one(fallback_text, task_description, label_schema)
                fallback["explanation"] = f"LLM failed, rule fallback: {fallback['explanation']}"
                return fallback
            raise LLMAgentError(self.last_error) from exc

    def predict_one(
        self,
        text: str,
        task_description: str,
        label_schema: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        return self._predict_one_prepared(text, text, task_description, label_schema)

    def predict_batch(
        self,
        texts: List[str],
        task_descriptions: List[str],
        records: Optional[List[Dict[str, object]]] = None,
        label_schema: Optional[Dict[str, str]] = None,
    ):
        records = records or [None] * len(texts)
        outputs = [
            self._predict_one_prepared(
                self.generate_input_text(text, record),
                text,
                task_description,
                _label_schema_for_sample(label_schema, task_description, record),
            )
            for text, task_description, record in zip(texts, task_descriptions, records)
        ]
        probs = np.stack([np.asarray(o["probs"], dtype=np.float32) for o in outputs], axis=0)
        confidences = np.array([o["confidence"] for o in outputs], dtype=np.float32)
        explanations = [str(o["explanation"]) for o in outputs]
        return probs, confidences, explanations


# clickbait task
# class LLMSemanticAgent(LLMBaseAgent):
#     name = "Semantic"
#     description = SemanticAgent.description
#     system_role = "语义上下文专家"
#     perspective = "你关注语义上下文、隐含意义、事实指向和文本整体含义。"
#     fewshot_file = "semantic.json"
#
#
# class LLMEmotionAgent(LLMBaseAgent):
#     name = "Emotion"
#     description = EmotionAgent.description
#     system_role = "情绪态度专家"
#     perspective = "你关注情绪、态度、主观色彩、讽刺和隐含感受。"
#     input_mode = "title"
#     fewshot_file = "emotion.json"
#
#
# class LLMIntentionAgent(LLMBaseAgent):
#     name = "Intention"
#     description = IntentionAgent.description
#     system_role = "点击诱导/操纵意图专家"
#     perspective = "你关注诱导点击、操纵意图、夸张承诺、误导性动机和劝服目的。"
#     input_mode = "title"
#     fewshot_file = "intention.json"
#
#
# class LLMLexicalAgent(LLMBaseAgent):
#     name = "Lexical"
#     description = LexicalAgent.description
#     system_role = "词汇表层模式专家"
#     perspective = "你关注表层词汇、标点、标题党模式、极端修饰和关键词线索。"
#     input_mode = "title"
#     fewshot_file = "lexical.json"
#
#
# class LLMConsistencyAgent(LLMBaseAgent):
#     name = "Consistency"
#     description = ConsistencyAgent.description
#     system_role = "一致性/矛盾/转折专家"
#     perspective = "你关注文本内部一致性、矛盾、转折、前后不匹配和叙述落差。"
#     fewshot_file = "consistency.json"

# implicit task
class LLMSemanticAgent(LLMBaseAgent):
    name = "Semantic"
    description = "语义隐含情感专家"
    system_role = "语义隐含情感专家"
    perspective = (
        "你关注文本整体语义、隐含意义、未直接表达的态度倾向，"
        "以及字面表达背后的真实情感。"
        "你的最终任务是判断文本是否包含隐式情感，而不是只判断语义是否通顺。"
    )
    fewshot_file = "implicit_semantic.json"


class LLMEmotionAgent(LLMBaseAgent):
    name = "Emotion"
    description = "情绪线索与情感倾向专家"
    system_role = "情绪线索与情感倾向专家"
    perspective = (
        "你关注文本中的情绪线索、评价性词语、语气词、标点、程度副词、"
        "隐性褒贬表达和局部情感触发信息。"
        "你的目标不是只寻找显性情绪词，而是判断这些线索是否共同指向隐式情感。"
    )
    fewshot_file = "emotion_clue.json"


class LLMIntentionAgent(LLMBaseAgent):
    name = "Intention"
    description = "语用意图与言外之意专家"
    system_role = "语用意图与言外之意专家"
    perspective = (
        "你关注文本是否通过语用方式间接表达情感，包括反问、讽刺、反语、委婉、夸张、"
        "言外之意、表面肯定但实际否定、表面中立但实际带有态度倾向等。"
        "你的目标不是单纯判断是否讽刺，而是判断这些表达是否指向隐含情感。"
    )
    fewshot_file = "pragmatic.json"


class LLMLexicalAgent(LLMBaseAgent):
    name = "Lexical"
    description = "语气词汇与表达线索专家"
    system_role = "语气词汇与表达线索专家"
    perspective = (
        "你关注文本中的表层表达线索，包括语气词、反问句式、感叹句式、重复标点、"
        "程度修饰词、网络化表达、隐性褒贬词和带有态度倾向的短语。"
        "你的目标不是简单统计词语，而是判断这些表达线索是否暗示隐式情感。"
    )
    fewshot_file = "lexical_emotion.json"


class LLMConsistencyAgent(LLMBaseAgent):
    name = "Consistency"
    description = "反差转折与隐含立场专家"
    system_role = "反差转折与隐含立场专家"
    perspective = (
        "你关注文本中的反差、转折、前后语义落差、表面表达与真实态度不一致、"
        "以及说话者对人物、事件、行为或观点的隐含立场。"
        "你的目标不是判断文本是否矛盾，而是通过反差和立场倾向推断是否存在隐式情感。"
    )
    fewshot_file = "stance_contrast.json"


# fake news task
# class LLMSemanticAgent(LLMBaseAgent):
#     name = "Semantic"
#     description = "新闻语义与事实主张专家"
#     system_role = "新闻语义与事实主张专家"
#     perspective = (
#         "你关注新闻文本的整体语义、核心事实主张、事件描述是否清晰可信，"
#         "以及文本中是否存在夸大、模糊、无法验证或逻辑上可疑的事实表述。"
#         "你的最终任务是判断该新闻是否包含虚构、误导性或难以核实的信息，"
#         "而不是只判断文本语义是否通顺。"
#     )
#     fewshot_file = "fake_semantic.json"
#
#
# class LLMEmotionAgent(LLMBaseAgent):
#     name = "Emotion"
#     description = "情绪煽动与误导性表达专家"
#     system_role = "情绪煽动与误导性表达专家"
#     perspective = (
#         "你关注新闻文本中的情绪化表达、煽动性措辞、强烈立场、夸张评价、"
#         "恐慌制造、愤怒引导、过度渲染和吸引眼球的表达方式。"
#         "你的目标不是单纯判断文本是否有情绪，而是判断这些情绪化线索是否可能服务于"
#         "误导读者、制造偏见或掩盖事实真实性。"
#     )
#     fewshot_file = "fake_emotion_clue.json"
#
#
# class LLMIntentionAgent(LLMBaseAgent):
#     name = "Intention"
#     description = "传播意图与误导策略专家"
#     system_role = "传播意图与误导策略专家"
#     perspective = (
#         "你关注新闻文本是否存在明显的误导性传播意图，包括断章取义、标题党式引导、"
#         "选择性呈现事实、偷换概念、诱导读者相信未经证实的信息、"
#         "将猜测包装成事实、或通过暗示方式制造错误认知。"
#         "你的目标不是判断作者主观动机本身，而是根据文本表现判断其是否具有假新闻或误导性信息特征。"
#     )
#     fewshot_file = "fake_pragmatic.json"
#
#
# class LLMLexicalAgent(LLMBaseAgent):
#     name = "Lexical"
#     description = "新闻措辞与可疑表达线索专家"
#     system_role = "新闻措辞与可疑表达线索专家"
#     perspective = (
#         "你关注新闻文本中的表层语言线索，包括绝对化用词、模糊来源、匿名爆料、"
#         "未经证实的说法、夸张标题、极端判断、模棱两可的事实描述、"
#         "以及缺少明确证据支持的断言。"
#         "你的目标不是简单统计可疑词语，而是判断这些表达线索是否共同指向虚假、误导或不可验证信息。"
#     )
#     fewshot_file = "fake_lexical.json"
#
#
# class LLMConsistencyAgent(LLMBaseAgent):
#     name = "Consistency"
#     description = "事实一致性与证据可信度专家"
#     system_role = "事实一致性与证据可信度专家"
#     perspective = (
#         "你关注新闻文本中的事实一致性、前后逻辑关系、事件因果关系、时间地点人物是否自洽，"
#         "以及文本中的结论是否有足够证据支撑。"
#         "你需要识别文本是否存在前后矛盾、证据不足、来源不明、事实跳跃、"
#         "以偏概全或将不确定信息表述为确定事实等问题。"
#         "你的目标不是单纯判断文本是否矛盾，而是通过一致性和证据可信度判断其是否可能是假新闻。"
#     )
#     fewshot_file = "fake_stance_contrast.json"


# traditional task
# class LLMSemanticAgent(LLMBaseAgent):
#     name = "Semantic"
#     description = "短文本语义理解专家"
#     system_role = "短文本语义理解专家"
#     perspective = (
#         "你关注短文本的整体语义、核心含义、主题指向和上下文中的隐含信息。"
#         "由于短文本通常信息量少、表达简略，你需要根据有限文本判断其最可能所属的类别。"
#         "你的最终任务是根据文本的主题、意图或语义含义进行分类，"
#         "而不是只判断文本是否通顺。"
#     )
#     fewshot_file = "short_semantic.json"
#
#
# class LLMEmotionAgent(LLMBaseAgent):
#     name = "Emotion"
#     description = "语气情绪与表达倾向专家"
#     system_role = "语气情绪与表达倾向专家"
#     perspective = (
#         "你关注短文本中的语气、情绪色彩、态度倾向、评价性表达、感叹、反问、"
#         "程度副词和具有主观色彩的词语。"
#         "你的目标不是单纯判断文本情绪，而是分析这些语气和情绪线索是否有助于区分文本类别，"
#         "例如判断文本更偏向询问、抱怨、推荐、评价、求助、新闻、娱乐或其他主题类别。"
#     )
#     fewshot_file = "short_emotion_clue.json"
#
#
# class LLMIntentionAgent(LLMBaseAgent):
#     name = "Intention"
#     description = "用户意图与交际目的专家"
#     system_role = "用户意图与交际目的专家"
#     perspective = (
#         "你关注短文本背后的用户意图和交际目的，包括询问、陈述、请求、推荐、吐槽、"
#         "分享、评价、求助、命令、提醒或表达观点等。"
#         "你的目标不是只看文本表面词语，而是判断说话者想通过这段短文本完成什么目的，"
#         "并根据意图信息辅助判断其所属类别。"
#     )
#     fewshot_file = "short_intention.json"
#
#
# class LLMLexicalAgent(LLMBaseAgent):
#     name = "Lexical"
#     description = "关键词与表层特征专家"
#     system_role = "关键词与表层特征专家"
#     perspective = (
#         "你关注短文本中的关键词、短语、实体名称、领域词、网络用语、特殊符号、"
#         "高频类别提示词和具有区分度的表层表达。"
#         "你的目标不是简单统计词语，而是根据这些关键词和表达线索判断文本更可能属于哪个主题、"
#         "意图或语义类别。"
#     )
#     fewshot_file = "short_lexical.json"
#
#
# class LLMConsistencyAgent(LLMBaseAgent):
#     name = "Consistency"
#     description = "类别一致性与判别边界专家"
#     system_role = "类别一致性与判别边界专家"
#     perspective = (
#         "你关注短文本与候选类别之间的一致性，判断文本内容、语义、意图和关键词是否共同指向同一类别。"
#         "当文本较短、信息不足或多个类别相似时，你需要分析不同类别之间的边界，"
#         "识别最符合文本含义的类别，并避免被单个关键词误导。"
#         "你的目标不是判断文本是否矛盾，而是判断文本与各类别定义之间的匹配程度。"
#     )
#     fewshot_file = "short_consistency.json"


class AgentFactory:
    rule_agent_classes = [SemanticAgent, EmotionAgent, IntentionAgent, LexicalAgent, ConsistencyAgent]
    llm_agent_classes = [
        LLMSemanticAgent,
        LLMEmotionAgent,
        LLMIntentionAgent,
        LLMLexicalAgent,
        LLMConsistencyAgent,
    ]

    @classmethod
    def build(cls, backend: Optional[str] = None) -> List[AgentInterface]:
        from config import AGENT_BACKEND, LLM_CACHE_ENABLED, LLM_CACHE_PATH
        from .cache import LLMCache
        from .llm_client import LLMClient

        backend = (backend or AGENT_BACKEND).strip().lower()
        rule_agents = [agent_cls() for agent_cls in cls.rule_agent_classes]
        if backend == "rule":
            return rule_agents
        if backend not in {"llm", "hybrid"}:
            raise ValueError(f"Unsupported AGENT_BACKEND={backend!r}; expected rule, llm, or hybrid.")

        llm_client = LLMClient.from_config()
        cache = LLMCache(LLM_CACHE_PATH) if LLM_CACHE_ENABLED else None
        fallback_agents = rule_agents if backend == "hybrid" else [None] * len(rule_agents)
        return [
            llm_cls(llm_client=llm_client, cache=cache, fallback_agent=fallback)
            for llm_cls, fallback in zip(cls.llm_agent_classes, fallback_agents)
        ]


def build_agents(backend: Optional[str] = None) -> List[AgentInterface]:
    return AgentFactory.build(backend)


def run_agents(
    agents: List[AgentInterface],
    texts: List[str],
    task_descriptions: List[str],
    records: Optional[List[Dict[str, object]]] = None,
    label_schema: Optional[Dict[str, str]] = None,
):
    probs, confidences, explanations = [], [], []
    for agent in agents:
        p, c, e = agent.predict_batch(
            texts,
            task_descriptions,
            records=records,
            label_schema=label_schema,
        )
        probs.append(p)
        confidences.append(c)
        explanations.append(e)
    agent_probs = np.stack(probs, axis=1)
    agent_confidences = np.stack(confidences, axis=1)
    return agent_probs.astype(np.float32), agent_confidences.astype(np.float32), explanations
