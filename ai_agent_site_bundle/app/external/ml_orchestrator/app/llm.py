from dataclasses import dataclass
from typing import Optional
from app.config import Cfg

PROMPT_TMPL = """Ты — помощник по тикетам поддержки/инженерным инцидентам.
Отвечай кратко и по делу. Если чего-то не хватает, явно укажи предположения.

Пользовательский запрос:
{query}

Контекст из базы знаний:
{context}

Сформируй ответ и краткий план действий в виде маркированного списка.
"""

@dataclass
class LLMResult:
    text: str

class LLMClient:
    def __init__(self):
        self.backend = Cfg.LLM_BACKEND
        if self.backend == "transformers":
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(Cfg.QWEN_MODEL, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                Cfg.QWEN_MODEL,
                device_map=Cfg.QWEN_DEVICE,
                torch_dtype="auto",
                trust_remote_code=True
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=Cfg.QWEN_MAX_NEW_TOKENS,
                temperature=Cfg.QWEN_TEMPERATURE,
                do_sample=False
            )
        elif self.backend == "openai":
            from openai import OpenAI
            self.client = OpenAI(base_url=Cfg.OPENAI_API_BASE, api_key=Cfg.OPENAI_API_KEY)
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    def generate(self, query: str, context: str) -> LLMResult:
        prompt = PROMPT_TMPL.format(query=query.strip(), context=context.strip() or "—")
        if self.backend == "transformers":
            out = self.pipe(prompt)[0]["generated_text"]
            gen = out[len(prompt):].strip() if out.startswith(prompt) else out.strip()
            return LLMResult(text=gen)
        else:
            resp = self.client.chat.completions.create(
                model=Cfg.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Ты краткий, точный ассистент по тикетам."},
                    {"role": "user", "content": prompt}
                ],
                temperature=Cfg.QWEN_TEMPERATURE,
                max_tokens=Cfg.QWEN_MAX_NEW_TOKENS
            )
            return LLMResult(text=resp.choices[0].message.content.strip())

def qwen_generate(query: str, context: str) -> str:
    return LLMClient().generate(query, context).text
