"""Извлечение сущностей и связей из чанка. Требование FR-1."""

from pydantic import BaseModel, ValidationError

from assistant.infra.config import vocabulary
from assistant.infra.grounding import canonical_name, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Chunk, Entity, Relation, ExtractionResult, EXTRACTION_SCHEMA, ENTITY_TYPES, RELATION_SIGNATURES

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
    relations = [Relation(subject=Entity(type=r.subject_type, name=r.subject), type=r.type, object=Entity(type=r.object_type, name=r.object))for r in data.relations]
    
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

                "Извлекай только явно упомянутые сущности и связи.\n"
                "Не додумывай факты и не используй внешние знания.\n\n"

                f"Допустимые типы сущностей: {entity_types}.\n\n"

                "Допустимые типы связей и направление:\n"
                f"{relations}\n\n"

                "Направление связи строгое:\n"
                "subject - источник, object - цель.\n\n"

                "Правила:\n"
                "1. Не создавай сущности, отсутствующие в тексте.\n"
                "2. Не создавай связи, которые нельзя подтвердить текстом.\n"
                "3. Типы subject и object должны соответствовать "
                "сигнатуре связи.\n"
                "4. Если информация неоднозначна, не создавай связь.\n"
                "5. Не меняй направление связи.\n"
                "6. Возвращай JSON строго по заданной схеме."
                "Пример: «Деталь P-1042 — пусковое реле. Вместо неё устанавливается P-1042B» "
                "даёт сущности Part:P-1042, Component:пусковое реле, Part:P-1042B "
                "и связи P-1042 IS_A пусковое реле, P-1042 REPLACED_BY P-1042B."
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
            llm_answer = llm.chat(messages=messages, json_schema=EXTRACTION_SCHEMA)
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

        result_entities.add(sub)
        result_entities.add(obj)

        result_relations.add(Relation(sub, relation.type, obj))

    result.entities = list(result_entities)
    result.relations = list(result_relations)
    return result
