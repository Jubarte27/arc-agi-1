"""
LLM Client Module for interacting with chat model endpoints (e.g., Gemma 31B IT).
"""

from typing import Dict, List
from . import config


def call_llm(
    messages: List[Dict[str, str]],
    model: str = config.MODEL_NAME,
    api_key: str = config.API_KEY,
    api_base_url: str = config.API_BASE_URL,
) -> str:
    """
    Sends chat messages to the LLM API and returns the generated text response.
    Supports google-genai, openai SDK, or direct HTTP request fallback.
    """
    # 1. Try Google Generative AI SDK
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role.capitalize()}:\n{content}\n")
        full_prompt = "\n".join(prompt_parts) + "\nAssistant:\n"
        
        response = client.models.generate_content(
            model=model,
            contents=full_prompt,
        )
        return response.text or ""
    except ImportError:
        pass
    except Exception as e:
        print(f"[Warning] google-genai client failed: {e}. Trying OpenAI-compatible client...")

    # 2. Try OpenAI SDK (for vLLM, OpenRouter, Together, Groq, Ollama, etc.)
    try:
        from openai import OpenAI
        client_kwargs = {"api_key": api_key}
        if api_base_url:
            client_kwargs["base_url"] = api_base_url
        client = OpenAI(**client_kwargs)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
    except ImportError:
        pass
    except Exception as e:
        print(f"[Warning] openai client failed: {e}. Trying HTTP requests fallback...")

    # 3. Fallback: Direct HTTP POST request
    try:
        import requests
        base = api_base_url.rstrip("/") if api_base_url else "https://api.openai.com/v1"
        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        }
        res = requests.post(url, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]
    except Exception as err:
        raise RuntimeError(
            f"Failed to communicate with LLM API. Please verify your API Key and endpoint. Error: {err}"
        )
