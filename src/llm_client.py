"""LLM client for generating grounded answers from retrieved data.
Supports OpenAI (default) and Anthropic (fallback when model name contains 'claude')."""

import os
from openai import OpenAI


def _is_anthropic_model(model: str) -> bool:
    return model and "claude" in model.lower()


def generate_answer(question: str, data_str: str, model: str = "gpt-4o") -> str:
    """Generate a natural language answer grounded in the retrieved data.
    Uses OpenAI by default; uses Anthropic when model contains 'claude' (e.g. fallback)."""
    system = """
    You are an answer-generation component in a Text-to-SQL system.

    You will receive:
    1) A user question
    2) The results of a SQL query that was generated specifically to answer that question

    IMPORTANT RULES:
    - The SQL query has already applied all filters and conditions that were explicitly required by the user question.
    - If the SQL result includes only selected columns, you MUST assume filtering was correctly enforced.
    - DO NOT require filtered columns to appear in the result in order to answer.

    ANSWERING GUIDELINES:
    - Base your answer strictly on the provided SQL results.
    - Do NOT invent or infer additional rows or values beyond the result set.
    - First determine whether the user question can be fully answered using ONLY the provided rows.
        - If yes, answer the question.
        - If no, clearly state what information is missing or ask a clarifying question.
    - If the result set is empty, state that no records match the criteria.
    - Do NOT say “insufficient data” merely because some columns are omitted.
    - Say “insufficient data” ONLY when the question logically requires information that cannot be derived from the rows.

    OUTPUT STYLE:
    - Be concise and factual.
    - Use natural language.
    - Format lists clearly.
    - Do not mention SQL, queries, or filtering logic.
    """
    user_content = f"""Question: {question}

Retrieved data:
{data_str}

Answer based on the data above:"""
    print('retrieved data')
    print(data_str)

    if _is_anthropic_model(model):
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        return (response.content[0].text if response.content else "").strip()

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()
