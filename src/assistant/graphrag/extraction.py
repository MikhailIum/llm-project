"""Извлечение сущностей и связей из чанка. Требование FR-1."""

from pydantic import BaseModel

from assistant.infra.config import vocabulary
from assistant.infra.grounding import canonical_name, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)


class EntitySchema(BaseModel):
    type: str
    name: str

class RelationSchema(BaseModel):
    subject_type: str
    subject: str
    type: str
    object_type: str
    object: str

class ExtractionSchema(BaseModel):
    entities: list[EntitySchema]
    relations: list[RelationSchema]

def parse(content):
    data = ExtractionSchema.model_validate_json(content)

    entities = [Entity(type=e.type, name=e.name) for e in data.entities]
    relations = [
        Relation(
            subject=Entity(type=r.subject_type, name=r.subject),
            type=r.type,
            object=Entity(type=r.object_type, name=r.object),
        )
        for r in data.relations
    ]

    return entities, relations

def validate_entity(entity, text):
    name = canonical_name(entity.type, entity.name, vocabulary())
    if name is None:
        return None

    if entity.type in {"Model", "Part", "ErrorCode", "Procedure"}:
        if name not in find_codes(text):
            return None
    else:
        if (entity.type, name) not in find_names(text, vocabulary()):
            return None

    return Entity(type=entity.type, name=name)

def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    entity_types = ", ".join(ENTITY_TYPES)

    relations = "\n".join(f"- {source} -{name}-> {target}" for name, (source, target) in RELATION_SIGNATURES.items())

    messages = [
        {
            "role": "system",
            "content": (
                "Ты - система извлечения сущностей и отношений из технической документации.\n\n"

                "Извлекай все сущности и связи, подтверждённые текстом. "
                "Связь может быть выражена прямо или следовать из структуры списка "
                "и однозначной формулировки; не требуй, чтобы тип связи был назван буквально.\n"
                "Не используй внешние знания и не добавляй связи между сущностями, "
                "которые текст не связывает.\n\n"

                f"Допустимые типы сущностей: {entity_types}.\n\n"

                "Procedure — это только код регламента вида SP-101. "
                "Шаги и инструкции, например «отключить оборудование», не являются сущностями.\n\n"

                "Допустимые типы связей и направление:\n"
                f"{relations}\n\n"

                "Направление связи строгое:\n"
                "subject - источник, object - цель.\n\n"

                "Правила:\n"
                "1. Не создавай сущности, отсутствующие в тексте или словаре.\n"
                "2. Извлекай все связи, которые прямо сказаны или однозначно следуют "
                "из текста; не добавляй связи по внешним знаниям.\n"
                "3. Для каждой позиции списка учитывай общий контекст перед списком, "
                "например модель, в которой установлены перечисленные детали.\n"
                "4. Типы subject и object должны соответствовать сигнатуре связи.\n"
                "5. Если связь неоднозначна или текст не связывает сущности, пропусти её.\n"
                "6. Не меняй направление связи.\n"
                "7. Возвращай JSON строго по заданной схеме.\n\n"
                "Примеры извлечения:\n"
                "А) В manual_AR-220.md сказано: «AR-220 входит в линейку „Арктика“. "
                "В AR-220 установлены следующие узлы: компрессор — "
                "деталь P-1010, поставщик „КомпрессорМаш“» подтверждает:\n"
                "- Model:AR-220 -BELONGS_TO-> ProductLine:Арктика\n"
                "- Model:AR-220 -HAS_PART-> Part:P-1010\n"
                "- Part:P-1010 -IS_A-> Component:компрессор\n"
                "- Part:P-1010 -SUPPLIED_BY-> Supplier:КомпрессорМаш\n"
                "Связь HAS_PART извлекается из общего контекста «в AR-220 установлены», "
                "а IS_A и SUPPLIED_BY — из описания этой позиции списка.\n"
                "Б) В manual_AR-220.md сказано: «E-18 — нет обдува испарителя, "
                "отказ узла „вентилятор испарителя“». Это подтверждает "
                "ErrorCode:E-18 -INDICATES-> Component:вентилятор испарителя.\n"
                "В) В procedure_SP-102.md сказано: «Регламент SP-102 применяется "
                "при ошибке E-12». Это подтверждает "
                "ErrorCode:E-12 -FIXED_BY-> Procedure:SP-102.\n"
                "Г) В report_R-2025-005.md оборудование BZ110, указан код E-18, "
                "а диагностика показала отказ узла «вентилятор испарителя». "
                "Это подтверждает Model:BZ-110 -FAILED-> "
                "Component:вентилятор испарителя.\n"
                "Д) В bulletin_SB-2025-01.md сказано: «Деталь P-1046 "
                "(вентилятор испарителя) снята с производства. Вместо неё "
                "устанавливается P-1046B». Это подтверждает "
                "Part:P-1046 -IS_A-> Component:вентилятор испарителя и "
                "Part:P-1046 -REPLACED_BY-> Part:P-1046B."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Текст чанка:\n{chunk.text}"
            ),
        },
    ]

    result = ExtractionResult(chunk.chunk_id, chunk.doc_id)
    for attempt in range(2):
        try:
            llm_answer = llm.chat(messages=messages, json_schema=EXTRACTION_SCHEMA, max_tokens=4096)
            entities, relations = parse(llm_answer.content)
            break
        except Exception as e:
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": llm_answer.content},
                    {
                        "role": "user",
                        "content": (
                            "Твой предыдущий ответ не соответствовал схеме JSON.\n"
                            f"Ошибка валидации: {e}\n\n"
                            "Исправь ответ с учётом ошибки. "
                            "Верни только корректный JSON, без пояснений."
                        ),
                    },
                ]
            else:
                return result

    result_entities = set()
    result_relations = set()

    for entity in entities:
        entity = validate_entity(entity, chunk.text)
        if entity is None:
            result.dropped += 1
        else:
            result_entities.add(entity)

    for relation in relations:
        sub = validate_entity(relation.subject, chunk.text)
        obj = validate_entity(relation.object, chunk.text)

        if sub is None or obj is None:
            result.dropped += 1
            continue

        relation = Relation(sub, relation.type, obj)
        if not relation.matches_schema():
            result.dropped += 1
            continue

        result_entities.add(sub)
        result_entities.add(obj)
        result_relations.add(relation)

    result.entities = list(result_entities)
    result.relations = list(result_relations)
    return result
