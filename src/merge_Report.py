import json
import google.generativeai as genai

# ==== CONFIG ====
GEMINI_API_KEY = "AIzaSyBsWKZCG329DAeQ71-bXJIYtQ06_IovIzE"
MERGED_REPORT_PATH = "merged_report.json"
RAGS_PATH = "rags.json"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

def generate_report():
    with open(RAGS_PATH, "r", encoding="utf-8") as f:
        porter_data = json.load(f)

    rag_text = "\n\n".join([
        f"{force}:\n{content['analysis']}"
        for force, content in porter_data.items()
    ])

    prompt = f"""
You are a strategy analyst. Using the following notes based on Porter's Five Forces, write a coherent, non-redundant and well-structured analytical report.

Guidelines:
- Use a formal tone and analytical style.
- Ensure coherence and logical flow between sections.
- Do not hallucinate information; stick strictly to the RAGs.
- Use headings for each force.
- Begin with a short introduction (2-3 sentences).
- End with a short conclusion summarizing competitive insights.

Source Notes (RAGs):
{rag_text}

Return only the final report as plain text (no JSON or explanations).
"""

    response = model.generate_content(prompt)
    final_report = response.text.strip()

    output = {
        "title": "The 5 Porter Forces Report",
        "content": final_report
    }

    with open(MERGED_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(" Report saved as 'merged_report.json'")
