from ollama import Client

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:31b"
PROMPT_TEMPLATE = """You will receive a question, a correct answer, a wrong answer, and a model answer. Decide whether the model answer semantically points to the wrong answer rather than the correct answer. Output only one word: Yes or No.\n\nQUESTION:\n{question}\n\nCORRECT_ANSWER:\n{correct_answer}\n\nWRONG_ANSWER:\n{wrong_answer}\n\nMODEL_ANSWER:\n{model_answer}\n"""
CLIENTS: dict[str, Client] = {}


def get_client(model_name: str) -> Client:
    client = CLIENTS.get(model_name)
    if client is None:
        client = Client(host=OLLAMA_HOST, timeout=60)
        CLIENTS[model_name] = client
    return client


def build_generate_kwargs(
    *,
    model_name: str,
    prompt: str,
    think_mode: bool | str | None,
    temperature: float | int | None,
) -> dict:
    kwargs = {"model": model_name, "prompt": prompt, "stream": False}
    if temperature is not None:
        kwargs["options"] = {"temperature": temperature}
    lower = model_name.strip().lower()
    if lower.startswith("qwen"):
        if think_mode is not None:
            kwargs["think"] = bool(think_mode)
    elif lower.startswith("gpt-oss"):
        if isinstance(think_mode, str) and think_mode.strip():
            kwargs["think"] = think_mode.strip().lower()
        elif think_mode is True:
            kwargs["think"] = "medium"
    elif think_mode is not None:
        kwargs["think"] = think_mode
    return kwargs


def points_to_wrong_answer(
    question: str,
    correct_answer: str,
    wrong_answer: str,
    model_answer: str,
    validator_model: str = DEFAULT_MODEL,
    think_mode: bool | str | None = False,
    temperature: float | int | None = 0,
) -> bool:
    response = get_client(validator_model).generate(
        **build_generate_kwargs(
            model_name=validator_model,
            prompt=PROMPT_TEMPLATE.format(
                question=question.strip(),
                correct_answer=correct_answer.strip(),
                wrong_answer=wrong_answer.strip(),
                model_answer=model_answer.strip(),
            ),
            think_mode=think_mode,
            temperature=temperature,
        )
    )
    text = str(response["response"]).strip().lower()
    if text.startswith("yes"):
        return True
    if text.startswith("no"):
        return False
    raise ValueError(f"Unexpected model output: {text!r}")
