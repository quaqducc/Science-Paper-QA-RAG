from typing import List

from transformers import pipeline

from .config import llm_cfg


class SimpleLLM:
    def __init__(self):
        # Prefer text2text-generation if the model supports it; fall back to text-generation
        try:
            self.pipe = pipeline("text2text-generation", model=llm_cfg.model, max_new_tokens=llm_cfg.max_new_tokens)
        except Exception:
            self.pipe = pipeline("text-generation", model=llm_cfg.model, max_new_tokens=llm_cfg.max_new_tokens)

    def answer(self, question: str, contexts: List[str]) -> str:
        prompt = (
            "You are a helpful scientific assistant. Use the provided paper abstracts to answer the question.\n"
            "Cite relevant paper IDs if helpful. If uncertain, say you don't know.\n\n"
            f"Question: {question}\n\n"
            "Contexts:\n" + "\n\n".join(contexts) + "\n\nAnswer:"
        )
        out = self.pipe(prompt, do_sample=False)[0]["generated_text"]
        return out[len(prompt):].strip() if out.startswith(prompt) else out


