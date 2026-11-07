"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def add_source(nodes, entity, chunk_id):
    if entity.node_id not in nodes.keys():
        nodes[entity.node_id] = set()
    nodes[entity.node_id].add(chunk_id)

def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph()
    graph.graph["conflicts"] = 0
    edges = {}
    nodes = {}    

    for result in results:
        for entity in result.entities:
            add_source(nodes, entity, result.chunk_id)

        for relation in result.relations:
            if not relation.matches_schema():
                continue
            sub = relation.subject
            obj = relation.object
            add_source(nodes, sub, result.chunk_id)
            add_source(nodes, obj, result.chunk_id)

    for result in results:
        for entity in result.entities:
            graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=nodes[entity.node_id])
        for relation in result.relations:
            if not relation.matches_schema():
                continue
            sub = relation.subject
            obj = relation.object
            graph.add_node(sub.node_id, type=sub.type, name=sub.name, sources=nodes[sub.node_id])
            graph.add_node(obj.node_id, type=obj.type, name=obj.name, sources=nodes[obj.node_id])
            if (sub.node_id, obj.node_id) not in edges.keys():
                edges[(sub.node_id, obj.node_id)] = set()
            edges[(sub.node_id, obj.node_id)].add(result.chunk_id)

    for result in results:
        for relation in result.relations:
            if not relation.matches_schema():
                continue
            sub = relation.subject
            obj = relation.object
            sources = edges[(sub.node_id, obj.node_id)]
            weight = len(sources)
            if (
                relation.type != "REPLACED_BY"
                or not graph.has_edge(obj.node_id, sub.node_id)
                or graph.edges[obj.node_id, sub.node_id]["type"] != "REPLACED_BY"
            ):
                graph.add_edge(sub.node_id, obj.node_id, type=relation.type, weight=weight, sources=sources)
            else:
                graph.graph["conflicts"] += 1
                sources_reversed = edges[(obj.node_id, sub.node_id)]
                weight_reversed = len(sources_reversed)
                if (weight > weight_reversed):
                    graph.remove_edge(obj.node_id, sub.node_id)
                    graph.add_edge(sub.node_id, obj.node_id, type=relation.type, weight=weight, sources=sources)
                elif weight_reversed == weight:
                    graph.remove_edge(obj.node_id, sub.node_id)

    return graph