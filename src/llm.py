import httpx

from load_env import openai_key, openai_model


def draft_answer(
    user_question: str, policies: list, playbooks: list, timeout: float = 60.0
) -> str:
    key = openai_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env.")

    system = (
        "You are a helpdesk coach for a fake company called Aurora Bean Roasters. "
        "Answer using ONLY the provided policy excerpts and playbooks. "
        "If something is not covered, say you do not have that in the knowledge base. "
        "Keep the reply short and use bullet points when helpful."
    )
    pol_block = "\n\n".join(
        f"Policy [{p.get('id')}] {p.get('title')}\n{p.get('text', '')[:1200]}"
        for p in policies
    )
    pb_block = "\n\n".join(
        f"Playbook [{p.get('id')}] {p.get('title')}\n{p.get('text', '')[:1200]}"
        for p in playbooks
    )
    user = f"Question:\n{user_question}\n\nPolicies:\n{pol_block}\n\nPlaybooks:\n{pb_block}"

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": openai_model(),
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, json=body)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    return data["choices"][0]["message"]["content"].strip()
