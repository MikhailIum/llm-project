"""Сообщества графа и их сводки. Требование FR-3."""

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    graph_wo_dirs = nx.Graph()
    graph_wo_dirs.add_nodes_from(graph.nodes)
    for sub, obj, attrs in graph.edges(data=True):
        if graph_wo_dirs.has_edge(sub, obj):
            graph_wo_dirs[sub][obj]["weight"] += attrs["weight"]
        else:
            graph_wo_dirs.add_edge(sub, obj, weight=attrs["weight"])

    communities = nx.community.louvain_communities(graph_wo_dirs, weight="weight", seed=7)
    return [Community(community_id, nodes) for community_id, nodes in enumerate(communities)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    entities = [f"- {node}: {graph.nodes[node]['type']} - {graph.nodes[node]['name']}" for node in community.nodes]

    relation_weights = {}
    entity_weights = {}
    edges = []
    for sub, obj, attrs in graph.edges(data=True):
        if sub not in community.nodes or obj not in community.nodes:
            continue
        relation_weights[attrs["type"]] = relation_weights.get(attrs["type"], 0) + attrs["weight"]
        entity_weights[sub] = entity_weights.get(sub, 0) + attrs["weight"]
        entity_weights[obj] = entity_weights.get(obj, 0) + attrs["weight"]
        edges.append(f"- {graph.nodes[sub]['name']} -{attrs['type']}-> {graph.nodes[obj]['name']} (вес связи: {attrs['weight']})")

    relation_stats = [f"- {relation_type}: {weight}" for relation_type, weight in relation_weights.items()]
    entity_stats = [f"- {graph.nodes[node]['name']}: {weight}" for node, weight in entity_weights.items()]
    
    facts = (
        f"Узлов: {len(community.nodes)}. Связей с учётом весов: {sum(relation_weights.values())}.\n"
        "Сущности:\n"
        + ("\n".join(entities) or "- нет")
        + "\nСвязи по типам (сумма весов):\n"
        + ("\n".join(relation_stats) or "- нет")
        + "\nВес связей, в которых участвует сущность:\n"
        + ("\n".join(entity_stats) or "- нет")
        + "\nСвязи сообщества:\n"
        + ("\n".join(sorted(edges)) or "- нет")
    )
    response = llm.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты формируешь краткие сводки сообществ графа знаний "
                    "на основе переданных структурированных данных.\n\n"

                    "Требования:\n"
                    "1. Опиши основные сущности сообщества и их типы.\n"
                    "2. Опиши связи между сущностями, сохраняя исходные "
                    "типы и направление отношений.\n"
                    "3. Выдели наиболее значимые сущности и типы связей, "
                    "ориентируясь исключительно на переданные веса.\n"
                    "4. Используй числовые значения только в том виде, "
                    "в котором они переданы.\n"
                    "5. Не делай выводов о причинах, последствиях и свойствах "
                    "сущностей, если они явно не представлены в данных.\n"
                    "6. Не добавляй сведения из внешних знаний.\n"
                    "7. Если данных недостаточно для содержательной сводки, "
                    "честно укажи это.\n\n"

                    "Не более 150 слов. Пиши информативно, без вводных фраз "
                    "и повторений."
                ),
            },
            {
                "role": "user",
                "content": f"Данные сообщества {community.community_id}:\n{facts}",
            },
        ]
    )
    return " ".join(response.content.split()[:150])
