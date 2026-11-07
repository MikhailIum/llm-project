"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Answer, KnowledgeBase, MAP_SCHEMA


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    codes = find_codes(question)
    names = find_names(question, kb.vocabulary)
    graph = kb.graph

    if any(f"{code_type(code)}:{code}" not in graph for code in codes):
        return Answer("Вопрос содержит код, которого нет в графе знаний.", refused=True)

    anchors = ([f"{code_type(code)}:{code}" for code in codes] + 
               [f"{kind}:{name}" for kind, name in names if f"{kind}:{name}" in graph])

    if not anchors:
        return Answer("Вопрос не содержит известных кодов или названий.", refused=True)

    selected_edges = set()
    if len(anchors) == 1:
        distances = nx.single_source_shortest_path_length(graph.to_undirected(), anchors[0], cutoff=2)
        for sub, obj in graph.edges:
            if sub in distances and obj in distances:
                selected_edges.add((sub, obj))
    else:
        for id, start in enumerate(anchors):
            for finish in anchors[id + 1 :]:
                for path in nx.all_simple_paths(graph.to_undirected(), start, finish, cutoff=3):
                    for left, right in zip(path, path[1:]):
                        if graph.has_edge(left, right):
                            selected_edges.add((left, right))
                        if graph.has_edge(right, left):
                            selected_edges.add((right, left))

    parts = set()
    for edge in selected_edges:
        for node in edge:
            if graph.nodes[node]["type"] == "Part":
                parts.add(node)

    while parts:
        part = parts.pop()
        for sub, obj, attrs in graph.out_edges(part, data=True):
            if attrs["type"] != "REPLACED_BY":
                continue
            selected_edges.add((sub, obj))
            parts.add(obj)

    facts = []
    chars = 0
    sources = set()
    for i, (sub, obj) in enumerate(selected_edges):
        line = f"{i}. {graph.nodes[sub]['name']} -{graph.edges[sub, obj]['type']}-> {graph.nodes[obj]['name']}"
        if chars <= 8000:
            facts.append(line)
            chars += len(line)
            sources.update(source.split("#", 1)[0] for source in graph.edges[sub, obj]["sources"])

    if len(facts) == 0:
        return Answer("В графе знаний не найдено достаточно фактов для ответа.", refused=True)

    facts_text = "\n".join(facts)
    response = llm.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "Отвечай на вопрос только по приведённым фактам графа. "
                    "Не добавляй сведения из внешних знаний. "
                    "Ответь прямо на то, о чём спрашивают: если нужен код детали, назови код; "
                    "если поставщик — назови поставщика; если регламент — назови его код. "
                    "Для вопроса о текущей детали пройди всю цепочку REPLACED_BY до последней детали. "
                    "Если спрашивают поставщика текущей детали, используй связь SUPPLIED_BY именно "
                    "для последней детали цепочки. Не подменяй связь наличия детали с моделью "
                    "связью FAILED (отказа)."
                ),
            },
            {"role": "user", "content": f"Вопрос: {question}\n\nФакты:\n{facts_text}"},
        ]
    )

    return Answer(response.content, sources=sorted(sources), facts=facts)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    part_answers = []
    sources = set()
    for community in kb.communities:
        response = llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ответь на вопрос только фактами, которые явно есть в этой сводке. "
                        "Верни JSON с полями answer (короткий ответ) и score (целое число от 0 до 10). "
                        "Поставь score 0, если сводка не отвечает на вопрос или не относится "
                        "к указанной в вопросе линейке/модели. Не считай общий вес сущности "
                        "частотой отказов: сравнивать частоту можно только по явно указанным "
                        "связям FAILED в нужной линейке. Не объединяй факты разных линеек."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Вопрос: {question}\n\nСводка сообщества:\n{community.summary}",
                },
            ],
            json_schema=MAP_SCHEMA,
        )
        try:
            partial = json.loads(response.content)
        except Exception:
            continue
        if not isinstance(partial, dict):
            continue
        answer = partial.get("answer")
        score = partial.get("score")
        if (not isinstance(answer, str) or not answer.strip() or not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 10):
            continue
        if score > 0:
            part_answers.append(f"Сообщество {community.community_id}: {answer.strip()}")
            for node in community.nodes:
                for source in kb.graph.nodes[node].get("sources", []):
                    sources.add(source.split("#", 1)[0])

    if not part_answers:
        return Answer("В графе знаний не найдено достаточно фактов для ответа.", refused=True)

    response = llm.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "Объедини частичные ответы в один точный ответ на вопрос. "
                    "Опирайся только на переданные ответы и не добавляй внешние сведения."
                ),
            },
            {
                "role": "user",
                "content": f"Вопрос: {question}\n\nЧастичные ответы:\n"
                + "\n".join(f"- {answer}" for answer in part_answers),
            },
        ]
    )
    return Answer(response.content, sources=list(sources))
