"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import code_type, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]

def function_schema(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }

QUESTION = {"type": "string", "description": "Вопрос пользователя"}
PART_ID = {"type": "string", "pattern": r"^P-\d{4}[A-Za-z]?$", "description": "Код детали"}
WAREHOUSE = {"type": "string", "enum": WAREHOUSES}

TOOL_SCHEMAS: list[dict] = [
    function_schema(
        "search_knowledge",
        "Найти конкретные факты в графе знаний по вопросу.",
        {"question": QUESTION},
        ["question"],
    ),
    function_schema(
        "summarize_topic",
        "Ответить на обзорный вопрос по сводкам сообществ графа.",
        {"question": QUESTION},
        ["question"],
    ),
    function_schema(
        "check_stock",
        "Проверить остаток детали на складе.",
        {"part_id": PART_ID, "warehouse": WAREHOUSE},
        ["part_id", "warehouse"],
    ),
    function_schema(
        "get_price",
        "Получить цену детали в рублях.",
        {"part_id": PART_ID},
        ["part_id"],
    ),
    function_schema(
        "create_service_order",
        "Создать заказ на обслуживание для детали и модели.",
        {
            "part_id": PART_ID,
            "warehouse": WAREHOUSE,
            "qty": {"type": "integer", "minimum": 1},
            "model": {"type": "string", "minLength": 1},
        },
        ["part_id", "warehouse", "qty", "model"],
    )
]

def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """

    def search_knowledge(question):
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize_topic(question):
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def validated_part_id(part_id):
        part_id = normalize_code(part_id)
        if part_id is None or code_type(part_id) != "Part":
            return None, "Некорректный номер детали: ожидается код вида P-..."
        if f"Part:{part_id}" not in kb.graph:
            return None, f"Деталь {part_id} отсутствует в графе знаний"
        return part_id, None

    def check_stock(part_id, warehouse):
        part_id, error = validated_part_id(part_id)
        if error:
            return {"error": error}

        if warehouse not in WAREHOUSES:
            return {"error": "Неизвестный склад"}

        try:
            return erp.get_stock(part_id, warehouse)
        except (ErpNotFound, ErpRejected) as e:
            if isinstance(e, ErpNotFound):
                return {"error": f"данные не найдены: {e}"}
            else:
                return {"error": f"операция отклонена ERP: {e}"}

    def get_price(part_id):
        part_id, error = validated_part_id(part_id)
        if error:
            return {"error": error}

        try:
            return erp.get_price(part_id)
        except (ErpNotFound, ErpRejected) as e:
            if isinstance(e, ErpNotFound):
                return {"error": f"данные не найдены: {e}"}
            else:
                return {"error": f"операция отклонена ERP: {e}"}

    def create_service_order(part_id, warehouse, qty, model):
        part_id, error = validated_part_id(part_id)
        if error:
            return {"error": error}

        if warehouse not in WAREHOUSES:
            return {"error": "Неизвестный склад"}

        model_code = normalize_code(model) if isinstance(model, str) else None
        if model_code is None or code_type(model_code) != "Model" or f"Model:{model_code}" not in kb.graph:
            return {"error": f"Модель {model} отсутствует в графе знаний"}

        if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
            return {"error": "Количество должно быть больше нуля"}

        try:
            return erp.create_order(part_id, warehouse, qty, model_code)
        except (ErpNotFound, ErpRejected) as e:
            if isinstance(e, ErpNotFound):
                return {"error": f"данные не найдены: {e}"}
            else:
                return {"error": f"операция отклонена ERP: {e}"}

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
