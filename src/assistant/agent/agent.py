"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.evaluation import fabricated_facts
from assistant.infra.grounding import find_codes, normalize_code
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep


def get_previous_codes(steps):
    codes = set()
    for step in steps:
        if not step.error and step.result:
            codes.update(find_codes(str(step.result)))
    return codes

def get_unsupported_facts(answer, question, steps):
    tool_results = [step.result for step in steps if not step.error and step.result is not None]
    return fabricated_facts(answer, question, get_previous_codes(steps), str(tool_results))

def get_sources(steps):
    sources = set()
    for step in steps:
        if not step.error:
            step_sources = step.result.get("sources", [])
            sources.update(step_sources)
    return sources


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    messages = [{"role": "user", "content": question}]
    steps = []
    refused_by_knowledge = False
    cached_results = {}

    while True:
        response = llm.chat(messages=messages, tools=TOOL_SCHEMAS)
        if not response.tool_calls:
            answer = response.content.strip()
            unsupported_facts = get_unsupported_facts(answer, question, steps)
            if unsupported_facts:
                correction_request = (
                    "В предыдущем ответе есть неподтверждённые коды, цены или остатки: "
                    + ", ".join(unsupported_facts)
                    + ". Сверь ответ только с вопросом и результатами инструментов, затем исправь его. "
                    "Не указывай факты, которых нет в этих данных."
                )
                messages.extend(
                    [
                        response.as_message(),
                        {"role": "user", "content": correction_request},
                    ]
                )
                corrected = llm.chat(messages=messages, tools=TOOL_SCHEMAS)
                corrected_answer = corrected.content.strip()
                if corrected.tool_calls or get_unsupported_facts(corrected_answer, question, steps):
                    return AgentResult(
                        "Не могу подтвердить эти сведения по имеющимся данным.",
                        steps=steps,
                        sources=get_sources(steps),
                        refused=True,
                    )
                answer = corrected_answer
            return AgentResult(
                answer,
                steps=steps,
                sources=get_sources(steps),
                refused=refused_by_knowledge,
            )

        messages.append(response.as_message())
        for call in response.tool_calls:
            if len(steps) >= max_steps:
                return AgentResult(
                    f"Работа прервана: достигнут лимит в {max_steps} вызовов инструментов. "
                    f"Успел получить: {str(steps)}",
                    steps=steps,
                    sources=get_sources(steps),
                )

            args = call.arguments
            if "_raw" in args:
                error = "Аргументы инструмента не удалось прочитать как JSON"
                steps.append(ToolStep(call.name, args, error=error))
                messages.append(tool_message(call, {"error": error}))
                continue

            function = tools.get(call.name)
            if function is None:
                error = f"Неизвестный инструмент: {call.name}"
                steps.append(ToolStep(call.name, args, error=error))
                messages.append(tool_message(call, {"error": error}))
                continue

            if call.name in {"check_stock", "get_price", "create_service_order"}:
                raw_part_id = args.get("part_id")
                part_id = normalize_code(raw_part_id) if isinstance(raw_part_id, str) else None
                previous_codes = get_previous_codes(steps)
                if part_id is None or part_id not in set(find_codes(question)) and part_id not in previous_codes:
                    error = (
                        f"номер детали {raw_part_id} не встречается в вопросе или результатах "
                        "предыдущих инструментов; сначала вызови search_knowledge, чтобы найти код детали"
                    )
                    steps.append(ToolStep(call.name, args, error=error))
                    messages.append(tool_message(call, {"error": error}))
                    continue
                args = {**args, "part_id": part_id}

            if call.name == "create_service_order":
                quantity = args.get("qty")
                question_quantities = {int(value) for value in re.compile(r"(?<![\w-])\d+(?![\w-])").findall(question)}
                if (not isinstance(quantity, int) or quantity <= 0 or quantity not in question_quantities):
                    prompt = "Сколько штук заказать? Укажи количество числом, прежде чем создавать заказ."
                    steps.append(ToolStep(call.name, args, error="количество не названо пользователем"))
                    return AgentResult(
                        prompt,
                        steps=steps,
                        sources=get_sources(steps),
                    )
                if not confirm:
                    steps.append(ToolStep(call.name, args, error="ожидается подтверждение пользователя"))
                    return AgentResult(
                        "Подтвердите создание заказа.",
                        steps=steps,
                        sources=get_sources(steps),
                        needs_confirmation=True,
                        pending_action={
                            "tool": call.name,
                            "args": args,
                        }
                    )

            call_key = json.dumps(
                {"tool": call.name, "args": args},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            if call_key in cached_results:
                result = cached_results[call_key]
                steps.append(ToolStep(call.name, args, result=result))
            else:
                try:
                    result = function(**args)
                except ErpUnavailable as first_error:
                    error_text = str(first_error)
                    for _ in range(2):
                        try:
                            result = function(**args)
                            break
                        except ErpUnavailable as retry_exc:
                            error_text = str(retry_exc)
                    else:
                        result = {"error": error_text}
                        steps.append(ToolStep(call.name, args, result=result, error=error_text))
                        messages.append(tool_message(call, result))
                        cached_results[call_key] = result
                        continue

                if not isinstance(result, dict):
                    result = {"error": "инструмент вернул результат неверного формата"}
                cached_results[call_key] = result
                if result.get("refused") is True and call.name in {"search_knowledge", "summarize_topic"}:
                    refused_by_knowledge = True
                steps.append(
                    ToolStep(
                        call.name,
                        args,
                        result=result,
                        error=result.get("error") if isinstance(result.get("error"), str) else None,
                    )
                )

            messages.append(tool_message(call, result))
