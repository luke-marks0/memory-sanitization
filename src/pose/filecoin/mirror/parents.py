from __future__ import annotations


def _parents(node: int, degree: int, nodes: int, stride: int) -> list[int]:
    if node <= 0:
        return []
    return [max(0, (node - ((offset + 1) * stride)) % nodes) for offset in range(degree)]


def drg_parents(node: int, degree: int, nodes: int) -> list[int]:
    return _parents(node=node, degree=degree, nodes=nodes, stride=1)


def expander_parents(node: int, degree: int, nodes: int) -> list[int]:
    return _parents(node=node, degree=degree, nodes=nodes, stride=3)

