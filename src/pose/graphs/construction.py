from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Iterable

from pose.common.errors import ProtocolError
from pose.graphs.descriptors import GraphDescriptor, build_graph_descriptor


@dataclass(frozen=True)
class PoseDbGraph:
    descriptor: GraphDescriptor
    predecessors: tuple[tuple[int, ...], ...]
    challenge_set: tuple[int, ...]

    @property
    def node_count(self) -> int:
        return len(self.predecessors)

    @property
    def label_count_m(self) -> int:
        return self.descriptor.label_count_m

    @property
    def graph_parameter_n(self) -> int:
        return self.descriptor.graph_parameter_n

    @property
    def gamma(self) -> int:
        return self.descriptor.gamma

    @property
    def hash_backend(self) -> str:
        return self.descriptor.hash_backend

    @property
    def label_width_bits(self) -> int:
        return self.descriptor.label_width_bits

    @property
    def graph_descriptor_digest(self) -> str:
        return self.descriptor.digest

    def challenge_node(self, challenge_index: int) -> int:
        if challenge_index < 0 or challenge_index >= len(self.challenge_set):
            raise ProtocolError(
                f"Challenge index {challenge_index} is outside [0, {len(self.challenge_set)})"
            )
        return self.challenge_set[challenge_index]

    def longest_path_lengths(self) -> tuple[int, ...]:
        lengths: list[int] = []
        for node_index, node_predecessors in enumerate(self.predecessors):
            if not node_predecessors:
                lengths.append(0)
                continue
            lengths.append(1 + max(lengths[predecessor] for predecessor in node_predecessors))
        return tuple(lengths)


@dataclass(frozen=True)
class _ConnectorComponent:
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]


@dataclass(frozen=True)
class _GraphComponent:
    level: int
    base: tuple[int, ...]
    challenge_set: tuple[int, ...]
    left: "_GraphComponent | None" = None
    center: _ConnectorComponent | None = None
    right: "_GraphComponent | None" = None


class _TempGraphBuilder:
    def __init__(self) -> None:
        self._next_node_id = 0
        self._edges: dict[int, set[int]] = {}

    def _new_node(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        self._edges.setdefault(node_id, set())
        return node_id

    def add_edge(self, source: int, target: int) -> None:
        if source == target:
            raise ProtocolError("Self-loops are not permitted in pose-db-drg-v1.")
        self._edges.setdefault(source, set()).add(target)
        self._edges.setdefault(target, set())

    def add_pairwise_edges(self, sources: Iterable[int], targets: Iterable[int]) -> None:
        source_nodes = tuple(sources)
        target_nodes = tuple(targets)
        if len(source_nodes) != len(target_nodes):
            raise ProtocolError(
                f"Pairwise edge lists must have the same size, got {len(source_nodes)} and {len(target_nodes)}"
            )
        for source, target in zip(source_nodes, target_nodes, strict=True):
            self.add_edge(source, target)

    def build_butterfly_connector(self, dimension: int) -> _ConnectorComponent:
        if dimension < 0:
            raise ProtocolError(f"Butterfly connector dimension must be non-negative, got {dimension}")
        width = 1 << dimension
        layers: list[list[int]] = []
        for _layer_index in range(dimension + 1):
            layers.append([self._new_node() for _ in range(width)])
        for layer_index in range(dimension):
            bit_index = dimension - 1 - layer_index
            for offset in range(width):
                self.add_edge(layers[layer_index][offset], layers[layer_index + 1][offset])
                self.add_edge(
                    layers[layer_index][offset],
                    layers[layer_index + 1][offset ^ (1 << bit_index)],
                )
        return _ConnectorComponent(
            inputs=tuple(layers[0]),
            outputs=tuple(layers[-1]),
        )

    def add_connector_between(self, inputs: tuple[int, ...], outputs: tuple[int, ...]) -> None:
        if not inputs:
            return
        if len(inputs) != len(outputs):
            raise ProtocolError(
                f"Connector endpoints must have the same size, got {len(inputs)} and {len(outputs)}"
            )
        if len(inputs) & (len(inputs) - 1):
            raise ProtocolError(
                f"Connector endpoints must have a power-of-two size, got {len(inputs)}"
            )
        connector = self.build_butterfly_connector(len(inputs).bit_length() - 1)
        self.add_pairwise_edges(inputs, connector.inputs)
        self.add_pairwise_edges(connector.outputs, outputs)

    def connect_operator(self, inputs: tuple[int, ...], component: _GraphComponent) -> None:
        if component.level == 0:
            self.add_pairwise_edges(inputs, component.base)
            return
        if component.left is None or component.center is None:
            raise ProtocolError("Malformed recursive graph component: missing left or center subcomponent.")
        midpoint = len(inputs) // 2
        self.connect_operator(inputs[:midpoint], component.left)
        self.add_connector_between(inputs[midpoint:], component.center.inputs)

    def build_recursive_graph(self, level: int) -> _GraphComponent:
        if level < 0:
            raise ProtocolError(f"Graph level must be non-negative, got {level}")
        if level == 0:
            node = self._new_node()
            return _GraphComponent(level=0, base=(node,), challenge_set=(node,))
        left = self.build_recursive_graph(level - 1)
        center = self.build_butterfly_connector(level - 1)
        right = self.build_recursive_graph(level - 1)
        self.add_pairwise_edges(left.base, center.inputs)
        self.connect_operator(center.outputs, right)
        return _GraphComponent(
            level=level,
            base=left.base + right.base,
            challenge_set=right.base,
            left=left,
            center=center,
            right=right,
        )

    def finalize(
        self,
        *,
        descriptor: GraphDescriptor,
        challenge_nodes: tuple[int, ...],
    ) -> PoseDbGraph:
        indegree = {node: 0 for node in self._edges}
        predecessors: dict[int, list[int]] = {node: [] for node in self._edges}
        for source, targets in self._edges.items():
            for target in targets:
                indegree[target] += 1
                predecessors[target].append(source)
        queue: list[int] = []
        for node, degree in indegree.items():
            if degree == 0:
                heappush(queue, node)
        topological: list[int] = []
        indegree_working = dict(indegree)
        adjacency = {node: sorted(targets) for node, targets in self._edges.items()}
        while queue:
            node = heappop(queue)
            topological.append(node)
            for target in adjacency[node]:
                indegree_working[target] -= 1
                if indegree_working[target] == 0:
                    heappush(queue, target)
        if len(topological) != len(self._edges):
            raise ProtocolError("pose-db-drg-v1 construction produced a cycle.")
        reindex = {old_node: new_node for new_node, old_node in enumerate(topological)}
        remapped_predecessors = []
        for old_node in topological:
            remapped_predecessors.append(
                tuple(sorted(reindex[predecessor] for predecessor in predecessors[old_node]))
            )
        return PoseDbGraph(
            descriptor=descriptor,
            predecessors=tuple(remapped_predecessors),
            challenge_set=tuple(reindex[node] for node in challenge_nodes),
        )


def build_pose_db_graph(
    *,
    label_count_m: int,
    graph_parameter_n: int | None = None,
    gamma: int | None = None,
    hash_backend: str,
    label_width_bits: int,
) -> PoseDbGraph:
    descriptor = build_graph_descriptor(
        label_count_m=label_count_m,
        graph_parameter_n=graph_parameter_n,
        gamma=gamma,
        hash_backend=hash_backend,
        label_width_bits=label_width_bits,
    )
    builder = _TempGraphBuilder()
    left_copy = builder.build_recursive_graph(descriptor.graph_parameter_n + 1)
    right_copy = builder.build_recursive_graph(descriptor.graph_parameter_n + 1)
    left_width = 1 << descriptor.graph_parameter_n
    retained_from_right = descriptor.label_count_m - left_width
    challenge_nodes = left_copy.challenge_set + right_copy.challenge_set[:retained_from_right]
    if len(challenge_nodes) != descriptor.label_count_m:
        raise ProtocolError(
            f"Challenge-set construction bug: expected {descriptor.label_count_m} nodes, got {len(challenge_nodes)}"
        )
    return builder.finalize(
        descriptor=descriptor,
        challenge_nodes=challenge_nodes,
    )
