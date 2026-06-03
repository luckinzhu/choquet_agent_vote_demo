import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_llm_api_key  # noqa: E402


DEFAULT_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]


def first_json_object(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
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
    raise ValueError("Unclosed JSON object")


def chat_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def model_candidates() -> list[str]:
    env_candidates = os.getenv("LLM_MODEL_CANDIDATES", "")
    values = [os.getenv("LLM_MODEL", DEFAULT_MODELS[0])]
    if env_candidates:
        values.extend(item.strip() for item in env_candidates.split(",") if item.strip())
    else:
        values.extend(DEFAULT_MODELS)
    deduped = []
    for item in values:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def call_gateway(base_url: str, model: str, api_key: str, timeout: int) -> tuple[int, str]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "只返回严格 JSON，不要 Markdown，不要额外解释。",
            },
            {
                "role": "user",
                "content": (
                    "请原样返回这个 JSON："
                    '{"class_0_probability":0.2,'
                    '"class_1_probability":0.8,'
                    '"confidence":0.9,'
                    '"explanation":"测试成功"}'
                ),
            },
        ],
        "temperature": 0,
    }
    request = urllib.request.Request(
        url=chat_url(base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def main() -> int:
    base_url = os.getenv("LLM_BASE_URL", "https://xiaohumini.site/v1")
    api_key_env = os.getenv("LLM_API_KEY_ENV", "LLM_API_KEY")
    api_key = get_llm_api_key()
    timeout = int(os.getenv("LLM_TIMEOUT", "60"))

    print(f"base_url: {base_url}")
    print(f"api_key_env: {api_key_env}")
    print(f"api_key_present: {bool(api_key)}")
    if not api_key:
        print("ERROR: missing API key. Set LLM_API_KEY in .env or environment.")
        return 2

    last_error = None
    for model in model_candidates():
        print(f"\nTrying model: {model}")
        try:
            status, body = call_gateway(base_url, model, api_key, timeout)
        except Exception as exc:
            last_error = str(exc)
            print("status: request failed")
            print(f"error: {exc}")
            continue

        print(f"status: {status}")
        print(f"raw response: {body}")
        if status < 200 or status >= 300:
            last_error = body[:500]
            continue
        try:
            response_json = json.loads(body)
            content = response_json["choices"][0]["message"]["content"]
            parsed = json.loads(first_json_object(content))
        except Exception as exc:
            last_error = f"response JSON parsing failed: {exc}"
            print(f"ERROR: {last_error}")
            continue

        print("parsed content:")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        print(f"SUCCESS_MODEL: {model}")
        return 0

    print("\nAll model attempts failed.")
    if last_error:
        print(f"last_error: {last_error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
