from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from functools import lru_cache
from heapq import heappop, heappush
from typing import Callable, Iterable, Iterator, Protocol

from pose.common.errors import ProtocolError
from pose.graphs.descriptors import GraphDescriptor, build_graph_descriptor

POSE_DB_GRAPH_CACHE_SIZE = 8


class _GraphTopology(Protocol):
    @property
    def node_count(self) -> int: ...

    @property
    def max_predecessor_count(self) -> int: ...

    def visit_predecessor_specs(
        self,
        consumer: Callable[[int, int, int], None],
    ) -> None: ...

    def visit_predecessors(self, consumer: Callable[[tuple[int, ...]], None]) -> None: ...

    def iter_predecessors(self) -> Iterator[tuple[int, ...]]: ...


@dataclass(frozen=True)
class _TupleGraphTopology:
    predecessor_rows: tuple[tuple[int, ...], ...]

    @property
    def node_count(self) -> int:
        return len(self.predecessor_rows)

    @property
    def max_predecessor_count(self) -> int:
        return max((len(row) for row in self.predecessor_rows), default=0)

    def visit_predecessor_specs(
        self,
        consumer: Callable[[int, int, int], None],
    ) -> None:
        for row in self.predecessor_rows:
            row_length = len(row)
            if row_length == 0:
                consumer(0, 0, 0)
            elif row_length == 1:
                consumer(1, row[0], 0)
            elif row_length == 2:
                consumer(2, row[0], row[1])
            else:
                raise ProtocolError(
                    "pose-db-drg-v1 predecessor arity exceeds the supported bound of 2."
                )

    def visit_predecessors(self, consumer: Callable[[tuple[int, ...]], None]) -> None:
        for row in self.predecessor_rows:
            consumer(row)

    def iter_predecessors(self) -> Iterator[tuple[int, ...]]:
        yield from self.predecessor_rows


@dataclass(slots=True)
class PoseDbGraph:
    descriptor: GraphDescriptor
    challenge_set: tuple[int, ...]
    _topology: _GraphTopology = field(repr=False, compare=False)
    _predecessor_rows_cache: tuple[tuple[int, ...], ...] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def node_count(self) -> int:
        return self._topology.node_count

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

    @property
    def predecessors(self) -> tuple[tuple[int, ...], ...]:
        if self._predecessor_rows_cache is None:
            self._predecessor_rows_cache = tuple(self.iter_predecessors())
        return self._predecessor_rows_cache

    @property
    def max_predecessor_count(self) -> int:
        return self._topology.max_predecessor_count

    def visit_predecessor_specs(
        self,
        consumer: Callable[[int, int, int], None],
    ) -> None:
        self._topology.visit_predecessor_specs(consumer)

    def visit_predecessors(self, consumer: Callable[[tuple[int, ...]], None]) -> None:
        self._topology.visit_predecessors(consumer)

    def iter_predecessors(self) -> Iterator[tuple[int, ...]]:
        yield from self._topology.iter_predecessors()

    def challenge_node(self, challenge_index: int) -> int:
        if challenge_index < 0 or challenge_index >= len(self.challenge_set):
            raise ProtocolError(
                f"Challenge index {challenge_index} is outside [0, {len(self.challenge_set)})"
            )
        return self.challenge_set[challenge_index]

    def longest_path_lengths(self) -> tuple[int, ...]:
        lengths: list[int] = []
        def consume(node_predecessors: tuple[int, ...]) -> None:
            node_index = len(lengths)
            if not node_predecessors:
                lengths.append(0)
                return
            lengths.append(1 + max(lengths[predecessor] for predecessor in node_predecessors))

        self.visit_predecessors(consume)
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
            challenge_set=tuple(reindex[node] for node in challenge_nodes),
            _topology=_TupleGraphTopology(tuple(remapped_predecessors)),
        )


@lru_cache(maxsize=None)
def _connector_node_count(dimension: int) -> int:
    if dimension < 0:
        raise ProtocolError(f"Butterfly connector dimension must be non-negative, got {dimension}")
    return (dimension + 1) * (1 << dimension)


@lru_cache(maxsize=None)
def _standalone_component_node_count(level: int) -> int:
    if level < 0:
        raise ProtocolError(f"Graph level must be non-negative, got {level}")
    if level == 0:
        return 1
    return (
        _standalone_component_node_count(level - 1)
        + _connector_node_count(level - 1)
        + _connected_component_node_count(level - 1)
    )


@lru_cache(maxsize=None)
def _connected_component_node_count(level: int) -> int:
    if level < 0:
        raise ProtocolError(f"Graph level must be non-negative, got {level}")
    if level == 0:
        return 1
    return (
        _connected_component_node_count(level - 1)
        + (2 * _connector_node_count(level - 1))
        + _connected_component_node_count(level - 1)
    )


def _connected_base_nodes(level: int, start: int) -> tuple[int, ...]:
    if level == 0:
        return (start,)
    right_start = start + _connected_component_node_count(level - 1) + (2 * _connector_node_count(level - 1))
    return _connected_base_nodes(level - 1, start) + _connected_base_nodes(level - 1, right_start)


def _standalone_base_nodes(level: int, start: int) -> tuple[int, ...]:
    if level == 0:
        return (start,)
    right_start = start + _standalone_component_node_count(level - 1) + _connector_node_count(level - 1)
    return _standalone_base_nodes(level - 1, start) + _connected_base_nodes(level - 1, right_start)


def _sorted_pair(first: int, second: int) -> tuple[int, int]:
    if first <= second:
        return (first, second)
    return (second, first)


class _FormulaDrivenEmitter:
    def __init__(
        self,
        spec_consumer: Callable[[int, int, int], None],
    ) -> None:
        self._next_node_index = 0
        self._spec_consumer = spec_consumer

    def _emit_row0(self) -> int:
        node_index = self._next_node_index
        self._next_node_index += 1
        self._spec_consumer(0, 0, 0)
        return node_index

    def _emit_row1(self, predecessor: int) -> int:
        node_index = self._next_node_index
        self._next_node_index += 1
        self._spec_consumer(1, predecessor, 0)
        return node_index

    def _emit_row2(self, first: int, second: int) -> int:
        node_index = self._next_node_index
        self._next_node_index += 1
        if first <= second:
            self._spec_consumer(2, first, second)
        else:
            self._spec_consumer(2, second, first)
        return node_index

    def _emit_connector(self, dimension: int, inputs: tuple[int, ...]) -> tuple[int, ...]:
        width = 1 << dimension
        if len(inputs) != width:
            raise ProtocolError(
                f"Connector input width mismatch: expected {width}, got {len(inputs)}"
            )
        previous_layer: list[int] = []
        for predecessor in inputs:
            previous_layer.append(self._emit_row1(predecessor))
        for layer_index in range(dimension):
            bit = 1 << (dimension - 1 - layer_index)
            current_layer: list[int] = []
            for offset in range(width):
                current_layer.append(self._emit_row2(previous_layer[offset], previous_layer[offset ^ bit]))
            previous_layer = current_layer
        return tuple(previous_layer)

    def _release_local_successor(
        self,
        local_successor: int,
        remaining_indegree: bytearray,
        ready_nodes: list[int],
    ) -> None:
        updated_indegree = remaining_indegree[local_successor] - 1
        remaining_indegree[local_successor] = updated_indegree
        if updated_indegree == 0:
            heappush(ready_nodes, local_successor)

    def _emit_merged_center_ingress(
        self,
        dimension: int,
        primary_inputs: tuple[int, ...],
        ingress_inputs: tuple[int, ...],
    ) -> tuple[int, ...]:
        width = 1 << dimension
        if len(primary_inputs) != width or len(ingress_inputs) != width:
            raise ProtocolError(
                "Merged connector input width mismatch: "
                f"expected {width}, got {len(primary_inputs)} and {len(ingress_inputs)}"
            )

        center_node_count = _connector_node_count(dimension)
        ingress_node_base = center_node_count
        total_local_nodes = center_node_count * 2
        remaining_indegree = bytearray(total_local_nodes)
        global_ids = array("Q", [0]) * total_local_nodes

        for offset in range(width):
            remaining_indegree[offset] = 1
        for layer in range(1, dimension + 1):
            layer_base = layer * width
            ingress_layer_base = ingress_node_base + layer_base
            for offset in range(width):
                remaining_indegree[layer_base + offset] = 2
                remaining_indegree[ingress_layer_base + offset] = 2

        ready_nodes = list(range(ingress_node_base, ingress_node_base + width))

        while ready_nodes:
            local_node = heappop(ready_nodes)
            if local_node < ingress_node_base:
                layer = local_node // width
                offset = local_node % width
                if layer == 0:
                    global_ids[local_node] = self._emit_row2(
                        primary_inputs[offset],
                        int(global_ids[ingress_node_base + (dimension * width) + offset]),
                    )
                else:
                    previous_layer_base = (layer - 1) * width
                    bit = 1 << (dimension - layer)
                    global_ids[local_node] = self._emit_row2(
                        int(global_ids[previous_layer_base + offset]),
                        int(global_ids[previous_layer_base + (offset ^ bit)]),
                    )
                if layer < dimension:
                    successor_layer_base = (layer + 1) * width
                    bit = 1 << (dimension - 1 - layer)
                    self._release_local_successor(
                        successor_layer_base + offset,
                        remaining_indegree,
                        ready_nodes,
                    )
                    self._release_local_successor(
                        successor_layer_base + (offset ^ bit),
                        remaining_indegree,
                        ready_nodes,
                    )
                continue

            ingress_local = local_node - ingress_node_base
            layer = ingress_local // width
            offset = ingress_local % width
            if layer == 0:
                global_ids[local_node] = self._emit_row1(ingress_inputs[offset])
            else:
                previous_layer_base = ingress_node_base + ((layer - 1) * width)
                bit = 1 << (dimension - layer)
                global_ids[local_node] = self._emit_row2(
                    int(global_ids[previous_layer_base + offset]),
                    int(global_ids[previous_layer_base + (offset ^ bit)]),
                )
            if layer < dimension:
                successor_layer_base = ingress_node_base + ((layer + 1) * width)
                bit = 1 << (dimension - 1 - layer)
                self._release_local_successor(
                    successor_layer_base + offset,
                    remaining_indegree,
                    ready_nodes,
                )
                self._release_local_successor(
                    successor_layer_base + (offset ^ bit),
                    remaining_indegree,
                    ready_nodes,
                )
            else:
                self._release_local_successor(
                    offset,
                    remaining_indegree,
                    ready_nodes,
                )

        return tuple(
            int(global_ids[(dimension * width) + offset])
            for offset in range(width)
        )

    def _emit_connected(self, level: int, inputs: tuple[int, ...]) -> tuple[int, ...]:
        width = 1 << level
        if len(inputs) != width:
            raise ProtocolError(
                f"Connected component input width mismatch: expected {width}, got {len(inputs)}"
            )
        if level == 0:
            return (self._emit_row1(inputs[0]),)
        half = len(inputs) // 2
        left_base = self._emit_connected(level - 1, inputs[:half])
        center_outputs = self._emit_merged_center_ingress(
            level - 1,
            left_base,
            inputs[half:],
        )
        right_base = self._emit_connected(level - 1, center_outputs)
        return left_base + right_base

    def _emit_standalone(self, level: int) -> tuple[int, ...]:
        if level == 0:
            return (self._emit_row0(),)
        left_base = self._emit_standalone(level - 1)
        center_outputs = self._emit_connector(level - 1, left_base)
        right_base = self._emit_connected(level - 1, center_outputs)
        return left_base + right_base

    def emit_graph(self, level: int) -> None:
        self._emit_standalone(level)
        self._emit_standalone(level)


@dataclass(frozen=True)
class _FormulaDrivenGraphTopology:
    descriptor: GraphDescriptor

    @property
    def node_count(self) -> int:
        level = self.descriptor.graph_parameter_n + 1
        return _standalone_component_node_count(level) * 2

    @property
    def max_predecessor_count(self) -> int:
        return 1 if self.descriptor.graph_parameter_n == 0 else 2

    def visit_predecessor_specs(
        self,
        consumer: Callable[[int, int, int], None],
    ) -> None:
        emitter = _FormulaDrivenEmitter(consumer)
        emitter.emit_graph(self.descriptor.graph_parameter_n + 1)

    def visit_predecessors(self, consumer: Callable[[tuple[int, ...]], None]) -> None:
        def consume_specs(predecessor_count: int, predecessor0: int, predecessor1: int) -> None:
            if predecessor_count == 0:
                consumer(())
            elif predecessor_count == 1:
                consumer((predecessor0,))
            else:
                consumer((predecessor0, predecessor1))

        self.visit_predecessor_specs(consume_specs)

    def iter_predecessors(self) -> Iterator[tuple[int, ...]]:
        rows: list[tuple[int, ...]] = []
        self.visit_predecessors(rows.append)
        yield from rows


def _build_pose_db_graph_uncached(descriptor: GraphDescriptor) -> PoseDbGraph:
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


def _build_pose_db_graph_formula(descriptor: GraphDescriptor) -> PoseDbGraph:
    level = descriptor.graph_parameter_n + 1
    left_copy_node_count = _standalone_component_node_count(level)
    left_copy_base = _standalone_base_nodes(level, 0)
    right_copy_base = _standalone_base_nodes(level, left_copy_node_count)
    half_width = len(left_copy_base) // 2
    retained_from_right = descriptor.label_count_m - (1 << descriptor.graph_parameter_n)
    challenge_nodes = left_copy_base[half_width:] + right_copy_base[half_width : half_width + retained_from_right]
    if len(challenge_nodes) != descriptor.label_count_m:
        raise ProtocolError(
            f"Challenge-set construction bug: expected {descriptor.label_count_m} nodes, got {len(challenge_nodes)}"
        )
    return PoseDbGraph(
        descriptor=descriptor,
        challenge_set=challenge_nodes,
        _topology=_FormulaDrivenGraphTopology(descriptor),
    )


@lru_cache(maxsize=POSE_DB_GRAPH_CACHE_SIZE)
def _build_pose_db_graph_cached(descriptor: GraphDescriptor) -> PoseDbGraph:
    return _build_pose_db_graph_formula(descriptor)


def clear_pose_db_graph_cache() -> None:
    _build_pose_db_graph_cached.cache_clear()


def pose_db_graph_cache_info():
    return _build_pose_db_graph_cached.cache_info()


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
    return _build_pose_db_graph_cached(descriptor)
