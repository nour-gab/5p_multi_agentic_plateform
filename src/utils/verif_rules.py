# finbert_utils.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

def load_finbert_model():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

def get_relevance_score(text, tokenizer, model):
    """
    Score based on confidence FinBERT assigns to 'positive' sentiment
    (indicating financial domain relevance).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    relevance_score = probs[0][1].item()  # index 1 = positive
    return relevance_score
# rules.py

import re

DIMENSIONS = {
    "Market Viability": ["demand", "target market", "problem", "solution", "pain point"],
    "Competitive Edge": ["unique", "differentiator", "advantage", "better than", "innovation"],
    "Revenue Model": ["subscription", "pricing", "fee", "commission", "monetize"],
    "Team Capability": ["team", "experience", "background", "skills", "track record"],
    "Relevance to FinTech": ["finance", "banking", "payments", "fintech", "investment"]
}

SUGGESTIONS = {
    "Market Viability": "Add specific target demographics or market pain points.",
    "Competitive Edge": "Highlight what makes your solution stand out.",
    "Revenue Model": "Include a monetization or pricing model.",
    "Team Capability": "Mention your team's skills or experience.",
    "Relevance to FinTech": "Clearly tie the idea to FinTech challenges or sectors."
}

def evaluate_idea(text: str) -> dict:
    """
    Returns a score [0–1] per dimension based on keyword heuristics.
    """
    text_lower = text.lower()
    results = {}

    for dimension, keywords in DIMENSIONS.items():
        matches = sum(1 for kw in keywords if re.search(rf'\b{re.escape(kw)}\b', text_lower))
        score = min(matches / len(keywords), 1.0)
        results[dimension] = {
            "score": round(score, 2),
            "keywords_found": [kw for kw in keywords if kw in text_lower]
        }

    return results

def suggest_enhancements(scores_dict: dict) -> list:
    """
    Suggest improvements based on weak dimensions.
    """
    suggestions = []
    for dimension, data in scores_dict.items():
        if data['score'] < 0.6 and dimension in SUGGESTIONS:
            suggestions.append({
                "area": dimension,
                "suggestion": SUGGESTIONS[dimension]
            })
    return suggestions
