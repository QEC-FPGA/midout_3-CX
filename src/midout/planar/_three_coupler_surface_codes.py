import functools

from typing import Any, Dict, FrozenSet, Tuple, Union, Literal, Optional

import stim

import random

from midout import gen

from midout._circuit_case import CircuitCase

from midout.gen._layer_translate import to_z_basis_interaction_circuit

_SKIP_DETECTOR = object()

def make_lattice_surgery_hex_planar_cx_code(
        **kwargs,
) -> CircuitCase:
    raise NotImplementedError(
        "LATTICE-SURGERY-3-CX is not a validated named operation. "
        "Use LATTICE-SURGERY-3-CX_MZZ with basis='Z' and detector_basis='Z'."
    )

def make_lattice_surgery_hex_planar_cx_mzz_code(
        *,
        basis: str,
        distance: int,
        rounds: int,
        Dm: int = None,
        detector_basis: Optional[Literal['X', 'Z']] = None,
        no_mid_reset: bool = False,
) -> CircuitCase:
    """Makes a first-pass three-coupler MZZ lattice surgery circuit.

    The circuit starts with two independent Z-basis memory patches, temporarily
    merges them through a one-row bridge, then splits them by X-measuring the
    bridge data qubits. Dm counts individual A or B extractions including the first pair; when omitted it
    defaults to the surrounding memory round count. Observable 0 is the MZZ
    value produced by the split measurement, while observables 1 and 2 are the
    final Z logicals of the two separated patches.
    """
    if no_mid_reset: raise ValueError("No-reset mode not supported by this short-merge package")
    assert rounds >= 2
    if Dm is None:
        Dm = rounds
    if basis != 'Z':
        raise NotImplementedError("LATTICE-SURGERY-3-CX_MZZ requires basis='Z'.")
    if detector_basis == 'X':
        raise NotImplementedError("LATTICE-SURGERY-3-CX_MZZ only supports Z-basis detectors.")
    detector_basis = 'Z'

    # Build the two separated memory patches plus a taller merged patch whose
    # middle row supplies the temporary lattice-surgery bridge.
    bridge_width = 1
    patch0 = ThreeCouplerLayoutHelper(
        basis=basis,
        distance=distance,
        width=distance,
        height=distance,
        origin=0,
        gate='CX',
        rounds=rounds,
        wiggle=False,
        detector_basis=detector_basis,
    )
    patch1 = ThreeCouplerLayoutHelper(
        basis=basis,
        distance=distance,
        width=distance,
        height=distance,
        origin=(distance + bridge_width) * 2j,
        gate='CX',
        rounds=rounds,
        wiggle=False,
        detector_basis=detector_basis,
    )
    merged = ThreeCouplerLayoutHelper(
        basis=basis,
        distance=distance,
        width=distance,
        height=distance * 2 + bridge_width,
        origin=0,
        gate='CX',
        rounds=max(2, Dm),
        wiggle=False,
        detector_basis=detector_basis,
    )

    builder = gen.Builder.for_qubits(merged.patch.data_set)
    syndrome_frame = _ThreeCouplerNoResetMeasurementFrame() if no_mid_reset else None
    memory_qubits = patch0.patch.data_set | patch1.patch.data_set
    bridge_used_qubits = merged.patch.data_set - memory_qubits
    memory_data_qubits = _layout_round_data(patch0) | _layout_round_data(patch1)
    bridge_data_qubits = _layout_round_data(merged) - memory_data_qubits

    # Initialize the two memory patches. In no-mid-reset mode the bridge is
    # also prepared up front because later syndrome measurements are not reset.
    bridge_aux_qubits = bridge_used_qubits - bridge_data_qubits
    if no_mid_reset:
        builder.gate("R", memory_qubits | (bridge_aux_qubits - merged.hadamard_qubits_A))
        builder.gate("RX", bridge_data_qubits | (bridge_aux_qubits & merged.hadamard_qubits_A))
    elif basis == 'X':
        builder.gate("RX", memory_qubits)
    else:
        builder.gate("R", memory_qubits)
    builder.tick()

    _three_coupler_measure_ab(
        layouts=[patch0, patch1],
        out=builder,
        cmp_a=None,
        cmp_b=None,
        save_a='mem_initA',
        save_b='mem_initB',
        syndrome_frame=syndrome_frame,
        measurement_scope_by_layout={id(patch0): ('memory', 0), id(patch1): ('memory', 1)} if no_mid_reset else None,
        reset_after_a_measure=not no_mid_reset,
        reset_after_b_measure=not no_mid_reset,
    )

    if rounds > 2:
        mem_loop = builder.fork()
        _three_coupler_round_b_to_b(
            layouts=[patch0, patch1],
            out=mem_loop,
            cmp_a='mem_initA',
            cmp_b='mem_initB',
            save_a='mem_loopA',
            save_b='mem_loopB',
            syndrome_frame=syndrome_frame,
            measurement_scope_by_layout={id(patch0): ('memory', 0), id(patch1): ('memory', 1)} if no_mid_reset else None,
            reset_after_a_measure=not no_mid_reset,
            reset_after_b_measure=not no_mid_reset,
        )
        builder.circuit += mem_loop.circuit * (rounds - 2)
        mem_a = 'mem_loopA'
        mem_b = 'mem_loopB'
    else:
        mem_a = 'mem_initA'
        mem_b = 'mem_initB'

    # Bring the bridge online and take the first merged-patch syndrome layer.
    # Detectors adjacent to the bridge need special comparison layers because
    # their previous measurements came from the two separated memory patches.
    if not no_mid_reset:
        builder.gate("R", bridge_aux_qubits - merged.hadamard_qubits_A)
        builder.gate("RX", bridge_data_qubits | (bridge_aux_qubits & merged.hadamard_qubits_A))
        builder.tick()

    _three_coupler_layer_type_BtoA_batched(layouts=[merged], out=builder)
    old_a_measure_qubits = patch0.measure_qubits_A | patch1.measure_qubits_A
    old_b_measure_qubits = patch0.measure_qubits_B | patch1.measure_qubits_B
    merged.make_circuit_layer_type_A_measure(
        out=builder,
        cmp_layer=_three_coupler_before_merge_cmp_layer(
            qs=merged.measure_qubits_A,
            old_qs=old_a_measure_qubits,
            cmp_layer=mem_a,
            save_layer='merge_initA',
            bridge_data_qubits=bridge_data_qubits,
        ),
        save_layer='merge_initA',
        no_mid_reset=no_mid_reset,
        syndrome_frame=syndrome_frame,
        measurement_scope=('merge',) if no_mid_reset else None,
    )
    _three_coupler_layer_type_AtoB_batched(layouts=[merged], out=builder)
    merged.make_circuit_layer_type_B_measure(
        out=builder,
        cmp_layer=_three_coupler_before_merge_cmp_layer(
            qs=merged.measure_qubits_B,
            old_qs=old_b_measure_qubits,
            cmp_layer=mem_b,
            save_layer='merge_initB',
            bridge_data_qubits=bridge_data_qubits,
        ),
        save_layer='merge_initB',
        no_mid_reset=no_mid_reset,
        syndrome_frame=syndrome_frame,
        measurement_scope=('merge',) if no_mid_reset else None,
    )

    # Dm now explicitly counts individual A or B extractions, including init A/B.
    # Odd Dm ends in A. No hidden merged B round is inserted.
    if Dm < 3 or Dm % 2 != 1:
        raise ValueError('Short-merge package requires an odd number >=3 of individual extractions')
    merge_a, merge_b = 'merge_initA', 'merge_initB'
    for j in range(2, Dm):
        phase = 'A' if j % 2 == 0 else 'B'
        if phase == 'A':
            _three_coupler_layer_type_BtoA_batched(layouts=[merged], out=builder)
        else:
            _three_coupler_layer_type_AtoB_batched(layouts=[merged], out=builder)
        name = f'merge_short_{j+1}_{phase}'
        _three_coupler_measure_layer_batched(
            layouts=[merged], out=builder, layer_type=phase,
            cmp_layer=merge_a if phase == 'A' else merge_b, save_layer=name,
            reset_after_measure=True)
        if phase == 'A': merge_a=name
        else: merge_b=name

    # Splitting is an X measurement of the bridge data. Its product is the MZZ
    # lattice-surgery outcome; the observable include is inserted after feedback
    # is inlined so it sits directly after this MX instruction.
    builder.measure(bridge_data_qubits, basis='X', save_layer='split')
    split_observable_targets = builder.tracker.current_measurement_record_targets_for(
        _three_coupler_lattice_surgery_mzz_split_observable_keys(
            merged=merged,
            bridge_data_qubits=bridge_data_qubits,
            split_prev_layer='A',
            merge_a=merge_a,
            merge_b=merge_b,
        )
    )
    builder.tick()
    # Last A syndrome was reset with feedback before split. Resume separated
    # patches in A -> B, not B -> A. Legacy interface detectors are not reused;
    # circuit_builder derives the full deterministic parity space independently.
    _three_coupler_layer_type_AtoB_batched(layouts=[patch0, patch1], out=builder)
    _three_coupler_measure_layer_batched(
        layouts=[patch0, patch1], out=builder, layer_type='B',
        cmp_layer={q:_SKIP_DETECTOR for q in old_b_measure_qubits},
        save_layer='split_resumeB', reset_after_measure=True)
    merge_b='split_resumeB'

    # Resume the two separated memory patches. The first post-split comparison
    # again needs bridge-aware references back into the merged-patch frame.
    _three_coupler_round_b_to_b(
        layouts=[patch0, patch1],
        out=builder,
        cmp_a=_three_coupler_after_split_cmp_layer(
            layout=merged,
            qs=old_a_measure_qubits,
            cmp_layer=merge_a,
            save_layer='after_initA',
            bridge_data_qubits=bridge_data_qubits,
        ),
        cmp_b=_three_coupler_after_split_cmp_layer(
            layout=merged,
            qs=old_b_measure_qubits,
            cmp_layer=merge_b,
            save_layer='after_initB',
            bridge_data_qubits=bridge_data_qubits,
        ),
        save_a='after_initA',
        save_b='after_initB',
        syndrome_frame=syndrome_frame,
        measurement_scope_by_layout={id(patch0): ('after_split', 0), id(patch1): ('after_split', 1)} if no_mid_reset else None,
        reset_after_a_measure=not no_mid_reset,
        reset_after_b_measure=not no_mid_reset,
    )

    if rounds > 2:
        after_loop = builder.fork()
        _three_coupler_round_b_to_b(
            layouts=[patch0, patch1],
            out=after_loop,
            cmp_a='after_initA',
            cmp_b='after_initB',
            save_a='after_loopA',
            save_b='after_loopB',
            syndrome_frame=syndrome_frame,
            measurement_scope_by_layout={id(patch0): ('after_split', 0), id(patch1): ('after_split', 1)} if no_mid_reset else None,
            reset_after_a_measure=not no_mid_reset,
            reset_after_b_measure=not no_mid_reset,
        )
        builder.circuit += after_loop.circuit * (rounds - 2)
        final_a = 'after_loopA'
        final_b = 'after_loopB'
    else:
        final_a = 'after_initA'
        final_b = 'after_initB'

    # Finish with ordinary memory readout of the two output patches. These are
    # tracked separately from the MZZ split result.
    _three_coupler_layer_type_B_end_batched(layouts=[patch0, patch1], out=builder)
    _three_coupler_final_measurement(
        layouts=[patch0, patch1],
        out=builder,
        cmp_a=final_a,
        cmp_b=final_b,
    )
    builder.obs_include([gen.AtLayer(q, "end") for q in _layout_final_observable_qubits(patch0)], obs_index=1)
    builder.obs_include([gen.AtLayer(q, "end") for q in _layout_final_observable_qubits(patch1)], obs_index=2)

    # The three-coupler layouts are built in the natural patch coordinates and
    # then transposed to match the display/orientation conventions used here.
    patch = merged.patch.after_coordinate_transform(lambda e: complex(e.imag, e.real))
    circuit = builder.circuit if no_mid_reset else builder.circuit.with_inlined_feedback()
    if not no_mid_reset:
        circuit = _three_coupler_insert_observable_after_measurement(
            circuit=circuit,
            measurement_name='MX',
            measurement_targets={builder.q2i[q] for q in bridge_data_qubits},
            observable_targets=split_observable_targets,
            obs_index=0,
        )
    circuit = gen.stim_circuit_with_transformed_coords(circuit, lambda e: complex(e.imag, e.real))
    return CircuitCase(
        circuit=circuit,
        patches=[patch],
        expected_interactions=frozenset(['CX']),
        round_end_data_qubit_sets=(),
    )

def _layout_round_data(layout: 'ThreeCouplerLayoutHelper') -> FrozenSet[complex]:
    return layout.patch.data_set - layout.measure_qubits_A - layout.measure_qubits_B

def _layout_final_observable_qubits(layout: 'ThreeCouplerLayoutHelper') -> FrozenSet[complex]:
    data = _layout_round_data(layout)
    if layout.basis == 'X':
        return frozenset(q for q in data if q.real == layout.origin.real)
    return frozenset(q for q in data if q.imag == layout.origin.imag)

def _three_coupler_lattice_surgery_mzz_split_observable_keys(
        *,
        merged: 'ThreeCouplerLayoutHelper',
        bridge_data_qubits: FrozenSet[complex],
        split_prev_layer: Literal['A', 'B'],
        merge_a: Any,
        merge_b: Any,
) -> list:
    """Returns the merged-boundary syndrome keys included in the split MZZ."""
    seam_y = _three_coupler_bridge_seam_y(bridge_data_qubits)
    if seam_y is None:
        return []
    max_bridge_x = max(q.real for q in bridge_data_qubits)
    if split_prev_layer == 'A':
        measure_qubits = merged.measure_qubits_A
        detector_basis = merged.detector_basis_at_layer_type_A
        layer = merge_a
    elif split_prev_layer == 'B':
        measure_qubits = merged.measure_qubits_B
        detector_basis = merged.detector_basis_at_layer_type_B
        layer = merge_b
    else:
        raise NotImplementedError(f'{split_prev_layer=}')

    return [
        gen.AtLayer(q, layer)
        for q in gen.sorted_complex(measure_qubits)
        if detector_basis(q) == 'Z'
        if abs(q.imag - seam_y) == 1
        if 0 <= q.real <= max_bridge_x + 1
    ]

def _three_coupler_insert_observable_after_measurement(
        *,
        circuit: stim.Circuit,
        measurement_name: str,
        measurement_targets: FrozenSet[int],
        observable_targets: list,
        obs_index: int,
) -> stim.Circuit:
    """Inserts an observable include immediately after a matching measurement."""
    if not observable_targets:
        return circuit
    result = stim.Circuit()
    inserted = False
    for instruction in circuit:
        result.append(instruction)
        if inserted:
            continue
        if isinstance(instruction, stim.CircuitRepeatBlock):
            continue
        if instruction.name != measurement_name:
            continue
        if {t.value for t in instruction.targets_copy()} != measurement_targets:
            continue
        result.append('OBSERVABLE_INCLUDE', observable_targets, obs_index)
        inserted = True
    if not inserted:
        raise ValueError(f'Failed to find {measurement_name=} with {measurement_targets=}.')
    return result

def _tile_is_type_a(tile: gen.Tile) -> bool:
    return (tile.measurement_qubit.real - tile.measurement_qubit.imag) % 4 == 1

def _three_coupler_cmp_layer_for(cmp_layer: Any, q: complex) -> Any:
    if isinstance(cmp_layer, dict):
        return cmp_layer.get(q)
    return cmp_layer

def _three_coupler_detector_keys_for_cmp_layer(
        *,
        prev_layer: Any,
        q: complex,
        save_layer: Any,
) -> Optional[list]:
    if prev_layer is _SKIP_DETECTOR:
        return None
    if prev_layer is None:
        return [gen.AtLayer(q, save_layer)]
    if isinstance(prev_layer, (list, tuple)):
        return [*prev_layer, gen.AtLayer(q, save_layer)]
    return [gen.AtLayer(q, prev_layer), gen.AtLayer(q, save_layer)]

def _three_coupler_after_split_cmp_layer(
        *,
        layout: 'ThreeCouplerLayoutHelper',
        qs: FrozenSet[complex],
        cmp_layer: Any,
        save_layer: Any,
        bridge_data_qubits: FrozenSet[complex],
) -> Dict[complex, list]:
    """Maps separated-patch syndrome qubits to their merged-frame references."""
    result = {}
    bridge_measure_qubits = (
        layout.measure_qubits_A | layout.measure_qubits_B
    ) - qs
    for q in gen.sorted_complex(qs):
        split_support = _three_coupler_verified_after_split_support_keys(
            layout=layout,
            q=q,
            cmp_layer=cmp_layer,
            bridge_data_qubits=bridge_data_qubits,
        )
        if split_support is not None:
            result[q] = split_support
        elif q + 2j in bridge_measure_qubits or q - 2j in bridge_measure_qubits:
            result[q] = [gen.AtLayer(q, cmp_layer)]
        else:
            result[q] = [gen.AtLayer(q, cmp_layer)]
    return result

def _three_coupler_verified_after_split_support_keys(
        *,
        layout: 'ThreeCouplerLayoutHelper',
        q: complex,
        cmp_layer: Any,
        bridge_data_qubits: FrozenSet[complex],
) -> Optional[list]:
    """Returns a verified four-coupler-style split support, when known.

    The four-coupler LS boundary detector uses:
        merge syndrome + split data support + after-split syndrome.

    In the three-coupler schedule the A/B syndrome qubits are not the same
    objects as the planar-code tile measurement qubits. The local tile-support
    candidates near the bridge therefore need explicit determinism validation
    before they are emitted. Returning None keeps the deterministic
    merge-syndrome + after-split-syndrome detector instead of writing an
    invalid split-data parity.
    """
    crossing_tiles = [
        tile
        for tile in layout.patch.tiles
        if q in tile.data_set
        if tile.data_set & bridge_data_qubits
    ]
    if not crossing_tiles:
        return None

    # No three-coupler split support has been validated yet. This helper is
    # intentionally structured around tile support so the first verified case
    # can be enabled without reintroducing coordinate-window heuristics.
    return None

def _three_coupler_before_merge_cmp_layer(
        *,
        qs: FrozenSet[complex],
        old_qs: FrozenSet[complex],
        cmp_layer: Any,
        save_layer: Any,
        bridge_data_qubits: FrozenSet[complex],
) -> Dict[complex, Any]:
    """Maps merged-patch syndrome qubits to valid pre-merge comparisons."""
    result = {}
    bridge_measure_qubits = qs - old_qs
    for q in gen.sorted_complex(qs):
        support = None
        if q in old_qs:
            support = _three_coupler_verified_before_merge_support(
                q=q,
                cmp_layer=cmp_layer,
                save_layer=save_layer,
                bridge_data_qubits=bridge_data_qubits,
            )
        if support is not None:
            result[q] = support
        elif q not in old_qs:
            result[q] = None if _three_coupler_verified_before_merge_current_only_detector(q=q) else _SKIP_DETECTOR
        elif q + 2j in bridge_measure_qubits or q - 2j in bridge_measure_qubits:
            result[q] = None if _three_coupler_verified_before_merge_current_only_detector(q=q) else _SKIP_DETECTOR
        else:
            result[q] = cmp_layer
    return result

def _three_coupler_verified_before_merge_support(
        *,
        q: complex,
        cmp_layer: Any,
        save_layer: Any,
        bridge_data_qubits: FrozenSet[complex],
) -> Optional[Any]:
    if save_layer not in ['merge_initA', 'merge_initB']:
        return None
    seam_y = _three_coupler_bridge_seam_y(bridge_data_qubits)
    if seam_y is None:
        return None
    distance = len(bridge_data_qubits)
    support = _three_coupler_even_before_merge_support(
        q=q,
        cmp_layer=cmp_layer,
        seam_y=seam_y,
        distance=distance,
    )
    if support is not None:
        return support
    if save_layer == 'merge_initA' and _three_coupler_before_merge_direct_detector_is_verified(
        q=q,
        seam_y=seam_y,
        distance=distance,
        layer_type='A',
    ):
        return cmp_layer
    if save_layer == 'merge_initB' and _three_coupler_before_merge_direct_detector_is_verified(
        q=q,
        seam_y=seam_y,
        distance=distance,
        layer_type='B',
    ):
        return cmp_layer
    if distance == 3 and save_layer == 'merge_initA' and q == 5 + 3j:
        previous_b_layer = _three_coupler_matching_b_layer(cmp_layer)
        if previous_b_layer is None:
            return None
        return [
            gen.AtLayer(3 + 3j, previous_b_layer),
            gen.AtLayer(5 + 3j, previous_b_layer),
        ]
    return None

def _three_coupler_even_before_merge_support(
        *,
        q: complex,
        cmp_layer: Any,
        seam_y: int,
        distance: int,
) -> Optional[list]:
    if distance % 2 == 1:
        return None
    if q.imag in [seam_y - 3, seam_y + 3]:
        return [gen.AtLayer(q, cmp_layer)]
    return None

def _three_coupler_bridge_seam_y(bridge_data_qubits: FrozenSet[complex]) -> Optional[int]:
    if not bridge_data_qubits:
        return None
    ys = {int(q.imag) for q in bridge_data_qubits}
    if len(ys) != 1:
        return None
    seam_y, = ys
    distance = len(bridge_data_qubits)
    expected = frozenset(2 * k + seam_y * 1j for k in range(distance))
    if bridge_data_qubits != expected:
        return None
    return seam_y

def _three_coupler_before_merge_direct_detector_is_verified(
        *,
        q: complex,
        seam_y: int,
        distance: int,
        layer_type: Literal['A', 'B'],
) -> bool:
    if distance % 2 == 0:
        return False
    if q.imag == seam_y - 3 and q.real == 2 * distance - 1:
        return True
    if q.imag == seam_y + 3:
        return True
    if q.imag == seam_y - 1:
        if layer_type == 'A':
            return q.real % 4 == 1
        if layer_type == 'B':
            return q.real % 4 == 3
    return False

def _three_coupler_matching_b_layer(a_layer: Any) -> Any:
    if a_layer == 'mem_loopA':
        return 'mem_loopB'
    if a_layer == 'mem_initA':
        return 'mem_initB'
    return None

def _three_coupler_verified_before_merge_current_only_detector(*, q: complex) -> bool:
    del q
    return False

def _three_coupler_reversed_transition_gate(k: str) -> str:
    if k == 'CXSWAP':
        return 'SWAPCX'
    if k == 'SWAPCX':
        return 'CXSWAP'
    return k

def _three_coupler_do_batched_transition(
        *,
        layers: Tuple[Tuple[Tuple[str, Any], ...], ...],
        out: gen.Builder,
) -> None:
    if not layers:
        return
    n = len(layers[0])
    assert all(len(layer) == n for layer in layers)
    for layer_index in range(n):
        by_gate = {}
        for transition in layers:
            k, p = transition[layer_index]
            by_gate.setdefault(k, []).extend(p)
        for k, pairs in by_gate.items():
            if k in ['CX', 'CXSWAP', 'SWAPCX']:
                out.gate2(k, pairs)
            else:
                out.gate(k, pairs)
        out.tick()

def _three_coupler_layer_type_AtoB_batched(
        *,
        layouts: list,
        out: gen.Builder,
) -> None:
    _three_coupler_do_batched_transition(
        layers=tuple(layout.transition_a_to_b for layout in layouts),
        out=out,
    )

def _three_coupler_layer_type_BtoA_batched(
        *,
        layouts: list,
        out: gen.Builder,
) -> None:
    _three_coupler_do_batched_transition(
        layers=tuple(tuple(
            (_three_coupler_reversed_transition_gate(k), p)
            for k, p in layout.transition_a_to_b[::-1]
        ) for layout in layouts),
        out=out,
    )

def _three_coupler_layer_type_B_end_batched(
        *,
        layouts: list,
        out: gen.Builder,
) -> None:
    if not layouts:
        return
    gate = layouts[0].gate
    assert all(layout.gate == gate for layout in layouts)
    if gate == 'CX':
        _three_coupler_layer_type_BtoA_batched(layouts=layouts, out=out)
    elif gate == 'CXSWAP':
        _three_coupler_do_batched_transition(
            layers=tuple((
                ('CXSWAP', layout.pairsB2),
                ('CXSWAP', layout.pairsB1),
            ) for layout in layouts),
            out=out,
        )
    elif gate == 'CX_MXX_MZZ':
        _three_coupler_do_batched_transition(
            layers=tuple((('CX', layout.pairsB1),) for layout in layouts),
            out=out,
        )
    else:
        raise NotImplementedError(f"{gate=}")

def _three_coupler_reset_syndrome_qubits_batched(
        *,
        layouts: list,
        out: gen.Builder,
        feedback_layer: Any = None,
) -> None:
    measure_qubits = frozenset().union(*(
        layout.measure_qubits_B for layout in layouts
    ))
    hadamard_qubits = frozenset().union(*(
        layout.measure_qubits_B & layout.hadamard_qubits_B
        for layout in layouts
    ))
    out.gate("R", measure_qubits - hadamard_qubits)
    out.gate("RX", measure_qubits & hadamard_qubits)
    if feedback_layer is not None:
        for layout in layouts:
            for m in gen.sorted_complex(layout.measure_qubits_B):
                out.classical_paulis(
                    control_keys=[gen.AtLayer(m, feedback_layer)],
                    targets=[m],
                    basis='Z' if m in layout.hadamard_qubits_B else 'X',
                )
    out.tick()

def _three_coupler_measure_layer_batched(
        *,
        layouts: list,
        out: gen.Builder,
        layer_type: Literal['A', 'B'],
        cmp_layer: Any,
        save_layer: Any,
        syndrome_frame: Optional[Any] = None,
        measurement_scope_by_layout: Optional[Dict[int, Any]] = None,
        reset_after_measure: bool = True,
) -> None:
    if not layouts:
        return
    if any(layout.gate not in ['CX', 'CXSWAP'] for layout in layouts):
        raise NotImplementedError(
            "Batched three-coupler LS measurement is only implemented for CX/CXSWAP-style layers."
        )

    def measure_qubits(layout: 'ThreeCouplerLayoutHelper') -> FrozenSet[complex]:
        return layout.measure_qubits_A if layer_type == 'A' else layout.measure_qubits_B

    def hadamard_qubits(layout: 'ThreeCouplerLayoutHelper') -> FrozenSet[complex]:
        return layout.hadamard_qubits_A if layer_type == 'A' else layout.hadamard_qubits_B

    def detector_basis(layout: 'ThreeCouplerLayoutHelper', q: complex) -> str:
        if layer_type == 'A':
            return layout.detector_basis_at_layer_type_A(q)
        return layout.detector_basis_at_layer_type_B(q)

    all_measure_qubits = frozenset().union(*(measure_qubits(layout) for layout in layouts))
    all_hadamard_qubits = frozenset().union(
        *(measure_qubits(layout) & hadamard_qubits(layout) for layout in layouts)
    )
    recorded_layer = save_layer if syndrome_frame is None else syndrome_frame.raw_layer(save_layer)
    out.measure(all_hadamard_qubits, save_layer=recorded_layer, basis='X')
    out.measure(all_measure_qubits - all_hadamard_qubits, save_layer=recorded_layer, basis='Z')
    if syndrome_frame is not None:
        for layout in layouts:
            measurement_scope = None if measurement_scope_by_layout is None else measurement_scope_by_layout[id(layout)]
            for q in gen.sorted_complex(measure_qubits(layout)):
                syndrome_frame.record_measurement(
                    tracker=out.tracker,
                    key=q,
                    layer=save_layer,
                    basis='X' if q in hadamard_qubits(layout) else 'Z',
                    scope=measurement_scope,
                )

    for layout in layouts:
        for m in gen.sorted_complex(measure_qubits(layout)):
            prev_layer = _three_coupler_cmp_layer_for(cmp_layer, m)
            basis = detector_basis(layout, m)
            detector_keys = _three_coupler_detector_keys_for_cmp_layer(
                prev_layer=prev_layer,
                q=m,
                save_layer=save_layer,
            )
            if detector_keys is None:
                continue
            if prev_layer is None:
                keep = basis == layout.basis
                keep &= layout.should_keep_detector_basis(basis)
                if keep:
                    out.detector(detector_keys, pos=m)
            elif layout.should_keep_detector_basis(basis):
                out.detector(detector_keys, pos=m)
    out.shift_coords(dt=1)
    out.tick()

    if syndrome_frame is None and reset_after_measure:
        reset_x_qubits = all_hadamard_qubits
        reset_z_qubits = all_measure_qubits - all_hadamard_qubits
        out.gate("R", reset_z_qubits)
        out.gate("RX", reset_x_qubits)
        for layout in layouts:
            for m in gen.sorted_complex(measure_qubits(layout)):
                out.classical_paulis(
                    control_keys=[gen.AtLayer(m, save_layer)],
                    targets=[m],
                    basis='Z' if m in hadamard_qubits(layout) else 'X',
                )
        out.tick()

def _three_coupler_measure_ab(
        *,
        layouts: list,
        out: gen.Builder,
        cmp_a: Any,
        cmp_b: Any,
        save_a: Any,
        save_b: Any,
        syndrome_frame: Optional[Any] = None,
        measurement_scope_by_layout: Optional[Dict[int, Any]] = None,
        reset_after_a_measure: bool = True,
        reset_after_b_measure: bool = True,
) -> None:
    _three_coupler_measure_layer_batched(
        layouts=layouts,
        out=out,
        layer_type='A',
        cmp_layer=cmp_a,
        save_layer=save_a,
        syndrome_frame=syndrome_frame,
        measurement_scope_by_layout=measurement_scope_by_layout,
        reset_after_measure=reset_after_a_measure,
    )
    _three_coupler_layer_type_AtoB_batched(layouts=layouts, out=out)
    _three_coupler_measure_layer_batched(
        layouts=layouts,
        out=out,
        layer_type='B',
        cmp_layer=cmp_b,
        save_layer=save_b,
        syndrome_frame=syndrome_frame,
        measurement_scope_by_layout=measurement_scope_by_layout,
        reset_after_measure=reset_after_b_measure,
    )

def _three_coupler_round_b_to_b(
        *,
        layouts: list,
        out: gen.Builder,
        cmp_a: Any,
        cmp_b: Any,
        save_a: Any,
        save_b: Any,
        syndrome_frame: Optional[Any] = None,
        measurement_scope_by_layout: Optional[Dict[int, Any]] = None,
        reset_after_a_measure: bool = True,
        reset_after_b_measure: bool = True,
) -> None:
    _three_coupler_layer_type_BtoA_batched(layouts=layouts, out=out)
    _three_coupler_measure_ab(
        layouts=layouts,
        out=out,
        cmp_a=cmp_a,
        cmp_b=cmp_b,
        save_a=save_a,
        save_b=save_b,
        syndrome_frame=syndrome_frame,
        measurement_scope_by_layout=measurement_scope_by_layout,
        reset_after_a_measure=reset_after_a_measure,
        reset_after_b_measure=reset_after_b_measure,
    )

def _three_coupler_final_measurement(
        *,
        layouts: list,
        out: gen.Builder,
        cmp_a: Any,
        cmp_b: Any,
) -> None:
    final_measurement_basis_by_qubit = {}
    for layout in layouts:
        final_measurement_basis_by_qubit.update(layout.final_measurement_basis_by_qubit())
    for final_basis in ['X', 'Z']:
        out.measure(
            [q for q, b in final_measurement_basis_by_qubit.items() if b == final_basis],
            save_layer="end",
            basis=final_basis,
        )
    for layout in layouts:
        for tile in layout.patch.tiles:
            if tile.basis != layout.basis:
                continue
            if not layout.should_keep_detector_basis(tile.basis):
                continue
            a_not_b = _tile_is_type_a(tile)
            offset = 1 if layout.basis == 'Z' else 1j
            if (a_not_b and layout.wiggle_a) or (not a_not_b and layout.wiggle_b):
                offset = -1j if layout.basis == 'Z' else -1
            if len(tile.data_set) == 1:
                d, = tile.data_set
                if d in layout.measure_qubits_A - layout.measure_qubits_B:
                    a_not_b = True
                    offset = 0
                elif d in layout.measure_qubits_B - layout.measure_qubits_A:
                    a_not_b = False
                    offset = 0
                elif d in layout.boundary_spike_qubits_z:
                    a_not_b = d in layout.odd_sums
                    offset = 0
                elif d in layout.boundary_spike_qubits_x:
                    a_not_b = d in layout.even_sums
                    offset = 0
            reference = gen.AtLayer(tile.measurement_qubit + offset, cmp_a if a_not_b else cmp_b)
            out.detector([
                             gen.AtLayer(d, "end") for d in layout.final_readout_qubits_for_end_tile(tile)
                         ] + [reference], pos=tile.measurement_qubit)

class _ThreeCouplerNoResetMeasurementFrame:
    """Tracks public syndrome layers through raw measurements without resets."""

    def raw_layer(self, layer: Any) -> Any:
        return 'raw', layer

    def record_measurement(
            self,
            *,
            tracker: gen.MeasurementTracker,
            key: complex,
            layer: Any,
            basis: str,
            scope: Any = None,
    ) -> None:
        raw_key = gen.AtLayer(key, self.raw_layer(layer))
        public_key = gen.AtLayer(key, layer)
        tracker.make_measurement_group([raw_key], key=public_key)

class ThreeCouplerLayoutHelper:
    """Helper class with useful properties for using three couplers."""
    def __init__(
            self,
            *,
            basis: str,
            distance: int,
            gate: str,
            rounds: int,
            wiggle: bool,
            detector_basis: Optional[Literal['X', 'Z']] = None,
            width: Optional[int] = None,
            height: Optional[int] = None,
            origin: complex = 0,
    ):
        assert rounds >= 2
        if detector_basis not in [None, 'X', 'Z']:
            raise ValueError(f"{detector_basis=}")
        self.basis = basis
        self.distance = distance
        self._default_square_layout = width is None and height is None and origin == 0
        self.width = distance if width is None else width
        self.height = distance if height is None else height
        self.origin = origin
        self.gate = gate
        self.rounds = rounds
        self.wiggle_a = False
        self.wiggle_b = wiggle
        self.detector_basis = detector_basis

    @property
    def use_more_spikes(self) -> bool:
        return self.gate == 'CXSWAP'

    @functools.cached_property
    def boundary_spike_qubits_x(self) -> FrozenSet[complex]:
        result = []
        for x in range(self.width - 1):
            result.append(self.origin + self.height * 2j + x * 2 - 1j + 1)
            if self.use_more_spikes:
                result.append(self.origin + x * 2 - 1j + 1)
        return frozenset(result)

    @functools.cached_property
    def boundary_spike_qubits_z(self) -> FrozenSet[complex]:
        if self._default_square_layout:
            return frozenset(q.imag + q.real * 1j for q in self.boundary_spike_qubits_x)
        result = []
        for y in range(self.height - 1):
            result.append(self.origin + (self.width * 2 - 1) + (y * 2 + 1) * 1j)
            if self.use_more_spikes:
                result.append(self.origin - 1 + (y * 2 + 1) * 1j)
        return frozenset(result)

    @functools.cached_property
    def boundary_spike_point_tiles(self) -> gen.Patch:
        return gen.Patch([
            *[
                gen.Tile(
                    bases="X",
                    measurement_qubit=c,
                    ordered_data_qubits=[c],
                ) for c in self.boundary_spike_qubits_x
            ],
            *[
                gen.Tile(
                    bases="Z",
                    measurement_qubit=c,
                    ordered_data_qubits=[c],
                ) for c in self.boundary_spike_qubits_z
            ],
        ])

    def is_tile_center(self, q: complex) -> bool:
        if self._default_square_layout:
            if q.real % 2 == q.imag % 2:
                return False
            if min(q.real, q.imag) < 0:
                return False
            return True
        q -= self.origin
        if q.real % 2 == q.imag % 2:
            return False
        if min(q.real, q.imag) < 0:
            return False
        if q.real > self.width * 2 - 1 or q.imag > self.height * 2 - 1:
            return False
        return True

    def is_data_qubit(self, q: complex) -> bool:
        if self._default_square_layout:
            return min(q.real, q.imag) >= 0 or q in self.boundary_spike_qubits_x | self.boundary_spike_qubits_z
        local = q - self.origin
        in_rect = (
            min(local.real, local.imag) >= 0
            and local.real <= self.width * 2 - 2
            and local.imag <= self.height * 2 - 2
        )
        return in_rect or q in self.boundary_spike_qubits_x | self.boundary_spike_qubits_z

    @functools.cached_property
    def patch(self) -> gen.Patch:
        adjacencies_x = [1j**d for d in range(4)][::-1]
        adjacencies_z = [1j**-d for d in range(3, 7)][::-1]

        tiles = [*self.boundary_spike_point_tiles.tiles]

        max_x = self.distance * 2 - 1 if self._default_square_layout else self.width * 2 - 1
        max_y = self.distance * 2 - 1 if self._default_square_layout else self.height * 2 - 1
        for x in range(-3, max_x):
            for y in range(-3, max_y):
                center = self.origin + x + y * 1j
                if not self.is_tile_center(center):
                    continue
                basis = "X" if center.real % 2 == 1 else 'Z'
                if (center.real - center.imag) % 4 == 1:
                    wiggle = self.wiggle_a
                else:
                    wiggle = self.wiggle_b
                if basis == 'X':
                    adjacencies = adjacencies_x
                else:
                    adjacencies = adjacencies_z
                if wiggle:
                    adjacencies = adjacencies[::-1]
                data_qubits = [
                    center + d
                    for d in adjacencies
                    if self.is_data_qubit(center + d)
                ]
                if len(data_qubits) < 2:
                    continue
                tiles.append(gen.Tile(
                    bases=basis,
                    measurement_qubit=center,
                    ordered_data_qubits=data_qubits,
                ))

        return gen.Patch(tiles)

    def should_keep_detector_basis(self, basis: str) -> bool:
        if self.detector_basis is None:
            return True
        return basis == self.detector_basis

    def detector_basis_at_layer_type_A(self, m: complex) -> str:
        if m in self.boundary_spike_qubits_z:
            return 'Z'
        if m in self.boundary_spike_qubits_x:
            return 'X'
        if (m in self.odd_sums) ^ self.wiggle_a:
            return 'X'
        return 'Z'

    def detector_basis_at_layer_type_B(self, m: complex) -> str:
        if m in self.boundary_spike_qubits_z:
            return 'Z'
        if m in self.boundary_spike_qubits_x:
            return 'X'
        if (m in self.odd_sums) ^ self.wiggle_b:
            return 'Z'
        return 'X'

    @functools.cached_property
    def even_sums(self) -> FrozenSet[complex]:
        return frozenset(
            q
            for q in self.patch.data_set
            if (q.real + q.imag) % 4 == 0
        )

    @functools.cached_property
    def odd_sums(self) -> FrozenSet[complex]:
        return frozenset(
            q
            for q in self.patch.data_set
            if (q.real + q.imag) % 4 == 2
        )

    @functools.cached_property
    def even_difs(self) -> FrozenSet[complex]:
        return frozenset(
            q
            for q in self.patch.data_set
            if (q.real - q.imag) % 4 == 0
        )

    @functools.cached_property
    def odd_difs(self) -> FrozenSet[complex]:
        return frozenset(
            q
            for q in self.patch.data_set
            if (q.real - q.imag) % 4 == 2
        )

    @functools.cached_property
    def pairsA1(self) -> FrozenSet[Tuple[complex, complex]]:
        return frozenset(
            (q, p)
            for q in self.patch.data_set
            if q in self.even_difs
            if (p := q + 1 - 1j) in self.patch.data_set
        )

    @functools.cached_property
    def pairsA2(self) -> FrozenSet[Tuple[complex, complex]]:
        return frozenset(
            (q, p)[::-1 if (q in self.even_difs) ^ self.wiggle_a else +1]
            for q in self.patch.data_set
            if q.real % 2 != 1
            if (p := q + 1 + 1j) in self.patch.data_set
            if (p in self.boundary_spike_qubits_z) <= (q in self.odd_sums)
            if (p in self.boundary_spike_qubits_x) <= (q in self.even_sums)
        )

    @property
    def start_backwards(self) -> bool:
        return self.rounds % 2 == 1

    def _cycle_sequence(
            self,
            *,
            three_coupler_start_cycle: Literal['auto', 'A', 'B'],
    ) -> Tuple[str, ...]:
        if three_coupler_start_cycle not in ['auto', 'A', 'B']:
            raise ValueError(f"{three_coupler_start_cycle=}")
        start = three_coupler_start_cycle
        if start == 'auto':
            start = 'B' if self.start_backwards else 'A'

        return tuple('AB'[(k + (0 if start == 'A' else 1)) % 2] for k in range(self.rounds))

    @functools.cached_property
    def measure_qubits_A(self) -> FrozenSet[complex]:
        result = frozenset(
            q
            for q in self.patch.data_set
            if q.real % 2 == 1
            and (q.imag != -1 or q.real % 4 == 3)
            and (q.real != -1 or q.imag % 4 == 1)
        )
        if self.wiggle_a:
            m = {}
            for a, b in self.pairsA2:
                m[a] = b
                m[b] = a
            result = frozenset(m.get(q, q) for q in result)

        return result

    @functools.cached_property
    def measure_qubits_B(self) -> FrozenSet[complex]:
        result = frozenset(
            q
            for q in self.patch.data_set
            if q.real % 2 == 1
            and (q.imag != -1 or q.real % 4 == 1)
            and (q.real != -1 or q.imag % 4 == 3)
        )
        if self.wiggle_b:
            m = {}
            for a, b in self.pairsB2:
                m[a] = b
                m[b] = a
            result = frozenset(m.get(q, q) for q in result)
        return result

    @functools.cached_property
    def hadamard_qubits_A(self) -> FrozenSet[complex]:
        return frozenset(
            q
            for q in self.patch.data_set
            if q in self.boundary_spike_qubits_x or (
                    q in self.measure_qubits_A
                    and q not in self.odd_difs
                    and (q not in self.even_sums) ^ self.wiggle_a
                    and q not in self.boundary_spike_qubits_z
            )
        )

    @functools.cached_property
    def pairsB1(self) -> FrozenSet[Tuple[complex, complex]]:
        return frozenset(
            (q, p)
            for q in self.patch.data_set
            if q in self.odd_difs
            if (p := q + 1 - 1j) in self.patch.data_set
        )

    @functools.cached_property
    def pairsB2(self) -> FrozenSet[Tuple[complex, complex]]:
        return frozenset(
            (q, p)[::-1 if (q in self.odd_difs) ^ self.wiggle_b else +1]
            for q in self.patch.data_set
            if q.real % 2 != 1
            if (p := q + 1 + 1j) in self.patch.data_set
            if (p in self.boundary_spike_qubits_z) <= (q in self.even_sums)
            if (p in self.boundary_spike_qubits_x) <= (q in self.odd_sums)
        )

    @functools.cached_property
    def hadamard_qubits_B(self) -> FrozenSet[complex]:
        return frozenset(
            q
            for q in self.patch.data_set
            if q in self.boundary_spike_qubits_x or (
                    q in self.measure_qubits_B
                    and q in self.odd_difs
                    and (q in self.even_sums) ^ self.wiggle_b
                    and q not in self.boundary_spike_qubits_z
            )
        )

    def make_circuit_layer_type_A_measure(
            self,
            *,
            out: gen.Builder,
            cmp_layer: Any,
            save_layer: Any,
            no_mid_reset: bool = False,
            syndrome_frame: Optional[_ThreeCouplerNoResetMeasurementFrame] = None,
            measurement_scope: Any = None) -> None:
        # Demolition measure the temporarily-single-qubit stabilizers.
        if self.gate == 'CX_MXX_MZZ':
            partner = {}
            for a, b in self.pairsA2:
                partner[a] = b
                partner[b] = a
            for q in self.measure_qubits_A:
                basis = 'X' if q in self.hadamard_qubits_A else 'Z'
                other = partner.get(q)
                targets = [q] if other is None else [q, other]
                if no_mid_reset:
                    assert syndrome_frame is not None
                    out.measure_pauli_product(
                        b2qs={basis: targets},
                        key=gen.AtLayer(q, syndrome_frame.raw_layer(save_layer)),
                    )
                    syndrome_frame.record_measurement(
                        tracker=out.tracker,
                        key=q,
                        layer=save_layer,
                        basis=basis,
                        scope=measurement_scope,
                    )
                else:
                    out.measure_pauli_product(b2qs={basis: targets}, key=gen.AtLayer(q, save_layer))
        elif self.gate in ['CX', 'CXSWAP']:
            if no_mid_reset:
                assert syndrome_frame is not None
                out.measure(
                    self.measure_qubits_A & self.hadamard_qubits_A,
                    save_layer=syndrome_frame.raw_layer(save_layer),
                    basis='X',
                )
                out.measure(
                    self.measure_qubits_A - self.hadamard_qubits_A,
                    save_layer=syndrome_frame.raw_layer(save_layer),
                    basis='Z',
                )
                for q in gen.sorted_complex(self.measure_qubits_A):
                    syndrome_frame.record_measurement(
                        tracker=out.tracker,
                        key=q,
                        layer=save_layer,
                        basis='X' if q in self.hadamard_qubits_A else 'Z',
                        scope=measurement_scope,
                    )
            else:
                out.measure(self.measure_qubits_A & self.hadamard_qubits_A, save_layer=save_layer, basis='X')
                out.measure(self.measure_qubits_A - self.hadamard_qubits_A, save_layer=save_layer, basis='Z')
        else:
            raise NotImplementedError(f'{self.gate=}')

        if cmp_layer is None:
            for m in gen.sorted_complex(self.measure_qubits_A):
                detector_basis = self.detector_basis_at_layer_type_A(m)
                keep = detector_basis == self.basis
                keep &= self.should_keep_detector_basis(detector_basis)
                if keep:
                    out.detector([gen.AtLayer(m, save_layer)], pos=m)
            out.shift_coords(dt=1)
        else:
            for m in gen.sorted_complex(self.measure_qubits_A):
                prev_layer = _three_coupler_cmp_layer_for(cmp_layer, m)
                detector_basis = self.detector_basis_at_layer_type_A(m)
                detector_keys = _three_coupler_detector_keys_for_cmp_layer(
                    prev_layer=prev_layer,
                    q=m,
                    save_layer=save_layer,
                )
                if detector_keys is None:
                    continue
                if prev_layer is None:
                    keep = detector_basis == self.basis
                    keep &= self.should_keep_detector_basis(detector_basis)
                    if keep:
                        out.detector(detector_keys, pos=m)
                elif self.should_keep_detector_basis(detector_basis):
                    out.detector(detector_keys, pos=m)
            out.shift_coords(dt=1)
        out.tick()
        if self.gate in ['CX', 'CXSWAP'] and not no_mid_reset:
            out.gate("R", self.measure_qubits_A - self.hadamard_qubits_A)
            out.gate("RX", self.measure_qubits_A & self.hadamard_qubits_A)
            for m in gen.sorted_complex(self.measure_qubits_A):
                out.classical_paulis(control_keys=[gen.AtLayer(m, save_layer)], targets=[m], basis='Z' if m in self.hadamard_qubits_A else 'X')
            out.tick()

    def make_circuit_layer_type_AtoB(self, *, out: gen.Builder) -> None:
        for k, p in self.transition_a_to_b:
            if k in ['CX', 'CXSWAP', 'SWAPCX']:
                out.gate2(k, p)
            else:
                out.gate(k, p)
            out.tick()

    def make_circuit_layer_type_BtoA(self, *, out: gen.Builder) -> None:
        for k, p in self.transition_a_to_b[::-1]:
            if k in ['CX', 'CXSWAP', 'SWAPCX']:
                if k == 'CXSWAP':
                    k = 'SWAPCX'
                elif k == 'SWAPCX':
                    k = 'CXSWAP'
                out.gate2(k, p)
            else:
                out.gate(k, p)
            out.tick()

    @functools.cached_property
    def transition_a_to_b(self) -> Tuple[Tuple[str, Any], ...]:
        if self.gate == 'CX':
            return (
                ("CX", self.pairsA2),
                ("CX", self.pairsA1),
                ("CX", self.pairsB1),
                ("CX", self.pairsB2),
            )
        elif self.gate == 'CXSWAP':
            return (
                ("CXSWAP", self.pairsA2),
                ("CXSWAP", self.pairsA1),
                ("SWAPCX", self.pairsB1),
                ("SWAPCX", self.pairsB2),
            )
        elif self.gate == 'CX_MXX_MZZ':
            return (
                ("CX", self.pairsA1),
                ("CX", self.pairsB1),
            )
        else:
            raise NotImplementedError(f"{self.gate=}")

    def make_circuit_layer_type_B_end(self, *, out: gen.Builder) -> None:
        # Undo interactions to restore the single qubit observables back to 4-body stabilizers.
        if self.gate == 'CX':
            self.make_circuit_layer_type_BtoA(out=out)
        elif self.gate == 'CXSWAP':
            out.gate2("CXSWAP", self.pairsB2)
            out.tick()
            out.gate2("CXSWAP", self.pairsB1)
            out.tick()
        elif self.gate == 'CX_MXX_MZZ':
            out.gate2("CX", self.pairsB1)
            out.tick()
        else:
            raise NotImplementedError(f"{self.gate=}")

    def make_circuit_layer_type_A_end(self, *, out: gen.Builder) -> None:
        if self.gate == 'CX':
            self.make_circuit_layer_type_AtoB(out=out)
            return
        raise NotImplementedError(f"{self.gate=}")

    def final_readout_qubits_for_end_tile(
            self,
            tile: gen.Tile,
            *,
            end_cycle: Literal['A', 'B'] = 'B',
        ) -> Tuple[complex, ...]:
        if self.gate != 'CX':
            return tile.ordered_data_qubits
        if end_cycle not in ['A', 'B']:
            raise ValueError(f"{end_cycle=}")
        if end_cycle == 'A':
            q2b = {q: tile.basis for q in tile.ordered_data_qubits}
            for pair_layer in [self.pairsB1, self.pairsB2]:
                for control, target in pair_layer:
                    if q2b.get(control) == 'X':
                        if target in q2b:
                            del q2b[target]
                        else:
                            q2b[target] = 'X'
                    if q2b.get(target) == 'Z':
                        if control in q2b:
                            del q2b[control]
                        else:
                            q2b[control] = 'Z'

            if any(b != tile.basis for b in q2b.values()):
                raise NotImplementedError(
                    f"Final three-coupler readout support included a basis other than {tile.basis!r}."
                )
            return tuple(gen.sorted_complex(q2b.keys()))

        q2b = {q: tile.basis for q in tile.ordered_data_qubits}
        for pair_layer in [self.pairsA1, self.pairsA2]:
            for control, target in pair_layer:
                if q2b.get(control) == 'X':
                    if target in q2b:
                        del q2b[target]
                    else:
                        q2b[target] = 'X'
                if q2b.get(target) == 'Z':
                    if control in q2b:
                        del q2b[control]
                    else:
                        q2b[control] = 'Z'

        if any(b != tile.basis for b in q2b.values()):
            raise NotImplementedError(
                f"Final three-coupler readout support included a basis other than {tile.basis!r}."
            )
        return tuple(gen.sorted_complex(q2b.keys()))

    def final_observable_qubits(self) -> FrozenSet[complex]:
        obs_xs = {q for q in self.patch.data_set if q.real == self.origin.real}
        obs_zs = {q for q in self.patch.data_set if q.imag == self.origin.imag}
        return frozenset(obs_xs if self.basis == 'X' else obs_zs)

    def final_measurement_basis_by_qubit(self, *, end_cycle: Literal['A', 'B'] = 'B') -> Dict[complex, str]:
        memory_readout_qubits = set(self.final_observable_qubits())
        for tile in self.patch.tiles:
            if tile.basis != self.basis:
                continue
            if not self.should_keep_detector_basis(tile.basis):
                continue
            memory_readout_qubits.update(self.final_readout_qubits_for_end_tile(tile, end_cycle=end_cycle))

        result = {}
        measure_qubits = self.measure_qubits_A | self.measure_qubits_B
        for q in self.patch.data_set:
            if q in memory_readout_qubits or q not in measure_qubits:
                result[q] = self.basis
            else:
                result[q] = 'X' if q in self.hadamard_qubits_A else 'Z'
        return result

    def make_circuit_layer_type_B_measure(
            self,
            *,
            out: gen.Builder,
            cmp_layer: Any,
            save_layer: Any,
            no_mid_reset: bool = False,
            syndrome_frame: Optional[_ThreeCouplerNoResetMeasurementFrame] = None,
            measurement_scope: Any = None) -> None:
        if self.gate in ['CX', 'CXSWAP']:
            if no_mid_reset:
                assert syndrome_frame is not None
                out.measure(
                    self.measure_qubits_B & self.hadamard_qubits_B,
                    save_layer=syndrome_frame.raw_layer(save_layer),
                    basis='X',
                )
                out.measure(
                    self.measure_qubits_B - self.hadamard_qubits_B,
                    save_layer=syndrome_frame.raw_layer(save_layer),
                    basis='Z',
                )
                for q in gen.sorted_complex(self.measure_qubits_B):
                    syndrome_frame.record_measurement(
                        tracker=out.tracker,
                        key=q,
                        layer=save_layer,
                        basis='X' if q in self.hadamard_qubits_B else 'Z',
                        scope=measurement_scope,
                    )
            else:
                out.measure(self.measure_qubits_B & self.hadamard_qubits_B, save_layer=save_layer, basis='X')
                out.measure(self.measure_qubits_B - self.hadamard_qubits_B, save_layer=save_layer, basis='Z')
        elif self.gate == 'CX_MXX_MZZ':
            partner = {}
            for a, b in self.pairsB2:
                partner[a] = b
                partner[b] = a
            for q in self.measure_qubits_B:
                basis = 'X' if q in self.hadamard_qubits_B else 'Z'
                other = partner.get(q)
                targets = [q] if other is None else [q, other]
                if no_mid_reset:
                    assert syndrome_frame is not None
                    out.measure_pauli_product(
                        b2qs={basis: targets},
                        key=gen.AtLayer(q, syndrome_frame.raw_layer(save_layer)),
                    )
                    syndrome_frame.record_measurement(
                        tracker=out.tracker,
                        key=q,
                        layer=save_layer,
                        basis=basis,
                        scope=measurement_scope,
                    )
                else:
                    out.measure_pauli_product(b2qs={basis: targets}, key=gen.AtLayer(q, save_layer))
        else:
            raise NotImplementedError(f'{self.gate=}')
        if cmp_layer is None:
            for m in gen.sorted_complex(self.measure_qubits_B):
                detector_basis = self.detector_basis_at_layer_type_B(m)
                keep = detector_basis == self.basis
                keep &= self.should_keep_detector_basis(detector_basis)
                if keep:
                    out.detector([gen.AtLayer(m, save_layer)], pos=m)
            out.shift_coords(dt=1)
        else:
            for m in gen.sorted_complex(self.measure_qubits_B):
                prev_layer = _three_coupler_cmp_layer_for(cmp_layer, m)
                detector_basis = self.detector_basis_at_layer_type_B(m)
                detector_keys = _three_coupler_detector_keys_for_cmp_layer(
                    prev_layer=prev_layer,
                    q=m,
                    save_layer=save_layer,
                )
                if detector_keys is None:
                    continue
                if prev_layer is None:
                    keep = detector_basis == self.basis
                    keep &= self.should_keep_detector_basis(detector_basis)
                    if keep:
                        out.detector(detector_keys, pos=m)
                elif self.should_keep_detector_basis(detector_basis):
                    out.detector(detector_keys, pos=m)
            out.shift_coords(dt=1)
        out.tick()
        if self.gate in ['CX', 'CXSWAP'] and not no_mid_reset:
            out.gate("R", self.measure_qubits_B - self.hadamard_qubits_B)
            out.gate("RX", self.measure_qubits_B & self.hadamard_qubits_B)
            for m in gen.sorted_complex(self.measure_qubits_B):
                out.classical_paulis(control_keys=[gen.AtLayer(m, save_layer)], targets=[m], basis='Z' if m in self.hadamard_qubits_B else 'X')
            out.tick()

    def make_ideal_circuit(
            self,
            *,
            no_mid_reset: bool = False,
            skip_first_measurement: bool = False,
            three_coupler_start_cycle: Literal['auto', 'A', 'B'] = 'auto',
    ) -> stim.Circuit:
        cycle_sequence = self._cycle_sequence(three_coupler_start_cycle=three_coupler_start_cycle)

        builder = gen.Builder.for_qubits(self.patch.data_set)
        syndrome_frame = _ThreeCouplerNoResetMeasurementFrame() if no_mid_reset else None

        # Prepare individual qubits into the basis needed for the very first interactions.
        x_init_qubits = set()
        if self.basis == 'X':
            x_init_qubits ^= self.patch.data_set
        builder.gate("RX", self.patch.data_set & x_init_qubits)
        builder.gate("R", self.patch.data_set - x_init_qubits)
        builder.tick()

        # Perform random Pauli gates to conduct quantum memory experiments on arbitrary quantum states
        # p = 0.1
        # print(self.patch.data_set)
        # if self.basis == 'X':
        #     z_qubits = set()
        #     for qubit in self.patch.data_set:
        #         random_float = random.random()
        #         if random_float < p:
        #             z_qubits.add(qubit)
        #     builder.gate("Z", z_qubits)
        # elif self.basis == 'Z':
        #     x_qubits = set()
        #     for qubit in self.patch.data_set:
        #         random_float = random.random()
        #         if random_float < p:
        #             x_qubits.add(qubit)
        #     builder.gate("X", x_qubits)
        # builder.tick()

        last_measurement_layer = {'A': None, 'B': None}
        saved_measurement_count = {'A': 0, 'B': 0}

        def emit_cycle(*, out: gen.Builder, k: int, save_layer: Any, cmp_layer: Any) -> None:
            cycle = cycle_sequence[k]
            if k:
                prev = cycle_sequence[k - 1]
                if prev == 'A' and cycle == 'B':
                    self.make_circuit_layer_type_AtoB(out=out)
                elif prev == 'B' and cycle == 'A':
                    self.make_circuit_layer_type_BtoA(out=out)
                else:
                    raise NotImplementedError(f"Bad three-coupler cycle transition: {prev!r} -> {cycle!r}.")

            if cycle == 'A':
                self.make_circuit_layer_type_A_measure(
                    out=out,
                    cmp_layer=cmp_layer,
                    save_layer=save_layer,
                    no_mid_reset=no_mid_reset,
                    syndrome_frame=syndrome_frame,
                )
            elif cycle == 'B':
                self.make_circuit_layer_type_B_measure(
                    out=out,
                    cmp_layer=cmp_layer,
                    save_layer=save_layer,
                    no_mid_reset=no_mid_reset,
                    syndrome_frame=syndrome_frame,
                )
            else:
                raise NotImplementedError(f"{cycle=}")

        def emit_explicit_cycle(k: int) -> None:
            cycle = cycle_sequence[k]
            n = saved_measurement_count[cycle]
            save_layer = f"init{cycle}" if n == 0 else f"loop{cycle}{n}"
            emit_cycle(
                out=builder,
                k=k,
                save_layer=save_layer,
                cmp_layer=last_measurement_layer[cycle],
            )
            last_measurement_layer[cycle] = save_layer
            saved_measurement_count[cycle] += 1

        k = 0
        while k < len(cycle_sequence):
            if skip_first_measurement and k == 0:
                k += 1
                continue
            if (
                    self.gate == 'CX'
                    and last_measurement_layer['A'] is not None
                    and last_measurement_layer['B'] is not None
            ):
                break
            emit_explicit_cycle(k)
            k += 1

        repeat_count = (len(cycle_sequence) - k) // 2
        if self.gate == 'CX' and repeat_count >= 2:
            loop = builder.fork()
            for j in range(2):
                cycle = cycle_sequence[k + j]
                emit_cycle(
                    out=loop,
                    k=k + j,
                    save_layer=f"loop{cycle}",
                    cmp_layer=last_measurement_layer[cycle],
                )
            builder.circuit += loop.circuit * repeat_count
            for j in range(2):
                cycle = cycle_sequence[k + j]
                last_measurement_layer[cycle] = f"loop{cycle}"
                saved_measurement_count[cycle] += repeat_count
            k += repeat_count * 2

        while k < len(cycle_sequence):
            if skip_first_measurement and k == 0:
                k += 1
                continue
            emit_explicit_cycle(k)
            k += 1

        final_cmp_a = last_measurement_layer['A']
        final_cmp_b = last_measurement_layer['B']
        if final_cmp_a is None or final_cmp_b is None:
            raise NotImplementedError(
                "The final three-coupler readout requires at least one measured A cycle and one measured B cycle."
            )

        # Rotate data qubits from the basis used for the last interaction to their measurement basis.
        end_cycle = cycle_sequence[-1]
        if end_cycle == 'A':
            self.make_circuit_layer_type_A_end(out=builder)
        elif end_cycle == 'B':
            self.make_circuit_layer_type_B_end(out=builder)
        else:
            raise NotImplementedError(f"{end_cycle=}")
        final_measurement_basis_by_qubit = self.final_measurement_basis_by_qubit(end_cycle=end_cycle)

        if self.gate != 'CX':
            for final_basis in ['X', 'Z']:
                builder.measure(
                    [q for q, b in final_measurement_basis_by_qubit.items() if b == final_basis],
                    save_layer="end",
                    basis=final_basis,
                )

            # Compare data qubit measurements to known stabilizer values from previous measurements.
            for tile in self.patch.tiles:
                if tile.basis != self.basis:
                    continue
                if not self.should_keep_detector_basis(tile.basis):
                    continue
                a_not_b = (tile.measurement_qubit.real - tile.measurement_qubit.imag) % 4 == 1
                offset = 1 if self.basis == 'Z' else 1j
                if (a_not_b and self.wiggle_a) or (not a_not_b and self.wiggle_b):
                    offset = -1j if self.basis == 'Z' else -1
                if len(tile.data_set) == 1:
                    d, = tile.data_set
                    if d in self.measure_qubits_A - self.measure_qubits_B:
                        a_not_b = True
                        offset = 0
                    elif d in self.measure_qubits_B - self.measure_qubits_A:
                        a_not_b = False
                        offset = 0
                    elif d in self.boundary_spike_qubits_z:
                        a_not_b = d in self.odd_sums
                        offset = 0
                    elif d in self.boundary_spike_qubits_x:
                        a_not_b = d in self.even_sums
                        offset = 0
                reference = gen.AtLayer(tile.measurement_qubit + offset, final_cmp_a if a_not_b else final_cmp_b)
                builder.detector([
                                     gen.AtLayer(d, "end")
                                     for d in self.final_readout_qubits_for_end_tile(tile, end_cycle=end_cycle)
                                 ] + [reference], pos=tile.measurement_qubit)

            # Extract observables from data measurements.
            builder.obs_include([gen.AtLayer(q, "end") for q in self.final_observable_qubits()], obs_index=0)

            if no_mid_reset:
                return builder.circuit
            return builder.circuit.with_inlined_feedback()

        def try_append_deterministic_detector(keys: list[gen.AtLayer], *, pos: complex) -> bool:
            coords = [pos.imag, pos.real, 0]
            targets = builder.tracker.current_measurement_record_targets_for(keys)
            trial = builder.circuit.copy()
            trial.append('DETECTOR', targets, coords)
            try:
                if no_mid_reset:
                    trial.detector_error_model(decompose_errors=True)
                else:
                    trial.with_inlined_feedback()
            except ValueError:
                return False
            builder.circuit.append('DETECTOR', targets, coords)
            return True

        final_syndrome_qubits = self.measure_qubits_A | self.measure_qubits_B
        if end_cycle in ['A', 'B']:
            terminal_hadamard_qubits = (
                self.hadamard_qubits_B
                if end_cycle == 'A'
                else self.hadamard_qubits_A
            )
            final_syndrome_basis_by_qubit = {
                q: 'X' if q in terminal_hadamard_qubits else 'Z'
                for q in final_syndrome_qubits
            }
            # Measure the final syndrome qubits first, preserving the same measurement bases
            # used by the terminal A/B readout frame.
            for final_basis in ['X', 'Z']:
                builder.measure(
                    [
                        q
                        for q, b in final_syndrome_basis_by_qubit.items()
                        if b == final_basis
                    ],
                    save_layer="end",
                    basis=final_basis,
                )

            # Compare the final syndrome readout against the previous same-type round.
            if end_cycle == 'B':
                for m in gen.sorted_complex(self.measure_qubits_A):
                    detector_basis = self.detector_basis_at_layer_type_A(m)
                    keep = detector_basis == self.basis
                    keep &= self.should_keep_detector_basis(detector_basis)
                    if keep:
                        builder.detector([gen.AtLayer(m, final_cmp_a), gen.AtLayer(m, "end")], pos=m)
            else:
                for m in gen.sorted_complex(self.measure_qubits_B):
                    detector_basis = self.detector_basis_at_layer_type_B(m)
                    keep = detector_basis == self.basis
                    keep &= self.should_keep_detector_basis(detector_basis)
                    if keep:
                        keys = [gen.AtLayer(m, final_cmp_b), gen.AtLayer(m, "end")]
                        try_append_deterministic_detector(keys, pos=m)
            builder.shift_coords(dt=1)

        # Now measure the round data qubits. Detectors using these measurements will appear
        # after the final syndrome readout for B-end, or directly after the last measured
        # syndrome cycle for A-end.
        round_data_qubits = self.patch.data_set - final_syndrome_qubits
        for final_basis in ['X', 'Z']:
            builder.measure(
                [
                    q
                    for q, b in final_measurement_basis_by_qubit.items()
                    if q in round_data_qubits
                    if b == final_basis
                ],
                save_layer="end",
                basis=final_basis,
            )
        final_data_measurement_indices = frozenset(
            builder.tracker.measurement_indices([
                gen.AtLayer(q, "end")
                for q in round_data_qubits
            ])
        )

        def final_data_detector_candidate(
                tile: gen.Tile,
                *,
                readout_end_cycle: Literal['A', 'B'],
                require_round_data: bool,
                expand_terminal_auxiliary: bool = False,
                extra_data_end_cycle: Optional[Literal['A', 'B']] = None,
        ) -> Optional[Tuple[list[gen.AtLayer], complex]]:
            readout_qubits = self.final_readout_qubits_for_end_tile(tile, end_cycle=readout_end_cycle)
            if require_round_data and not (set(readout_qubits) & round_data_qubits):
                return None
            a_not_b = (tile.measurement_qubit.real - tile.measurement_qubit.imag) % 4 == 1
            offset = 1 if self.basis == 'Z' else 1j
            if (a_not_b and self.wiggle_a) or (not a_not_b and self.wiggle_b):
                offset = -1j if self.basis == 'Z' else -1
            if len(tile.data_set) == 1:
                d, = tile.data_set
                if d in self.measure_qubits_A - self.measure_qubits_B:
                    a_not_b = True
                    offset = 0
                elif d in self.measure_qubits_B - self.measure_qubits_A:
                    a_not_b = False
                    offset = 0
                elif d in self.boundary_spike_qubits_z:
                    a_not_b = d in self.odd_sums
                    offset = 0
                elif d in self.boundary_spike_qubits_x:
                    a_not_b = d in self.even_sums
                    offset = 0
            reference = gen.AtLayer(tile.measurement_qubit + offset, final_cmp_a if a_not_b else final_cmp_b)
            keys = []
            for d in readout_qubits:
                if expand_terminal_auxiliary and d in final_syndrome_qubits:
                    keys.append(gen.AtLayer(d, final_cmp_b))
                    keys.append(gen.AtLayer(d, "end"))
                else:
                    keys.append(gen.AtLayer(d, "end"))
            if extra_data_end_cycle is not None:
                for d in self.final_readout_qubits_for_end_tile(tile, end_cycle=extra_data_end_cycle):
                    if d in round_data_qubits:
                        keys.append(gen.AtLayer(d, "end"))
            keys.append(reference)
            return keys, reference.key

        def has_final_data_measurement_key(keys: list[gen.AtLayer]) -> bool:
            return bool(
                set(builder.tracker.measurement_indices(keys))
                & final_data_measurement_indices
            )

        emitted_a_end_data_detector_keys = set()

        # Compare data qubit measurements to known stabilizer values from previous measurements.
        for tile in self.patch.tiles:
            if tile.basis != self.basis:
                continue
            if not self.should_keep_detector_basis(tile.basis):
                continue
            if end_cycle == 'B':
                candidate = final_data_detector_candidate(tile, readout_end_cycle=end_cycle, require_round_data=True)
                if candidate is None:
                    continue
                keys, reference_pos = candidate
                builder.detector(keys, pos=reference_pos)
            else:
                readout_candidate_styles = [
                    ('A', True, False, None),
                ]
                for (
                        readout_end_cycle,
                        require_round_data,
                        expand_terminal_auxiliary,
                        extra_data_end_cycle,
                ) in readout_candidate_styles:
                    candidate = final_data_detector_candidate(
                        tile,
                        readout_end_cycle=readout_end_cycle,
                        require_round_data=require_round_data,
                        expand_terminal_auxiliary=expand_terminal_auxiliary,
                        extra_data_end_cycle=extra_data_end_cycle,
                    )
                    if candidate is None:
                        continue
                    keys, reference_pos = candidate
                    if not has_final_data_measurement_key(keys):
                        continue
                    key_id = tuple(keys)
                    if key_id in emitted_a_end_data_detector_keys:
                        continue
                    emitted_a_end_data_detector_keys.add(key_id)
                    try_append_deterministic_detector(keys, pos=reference_pos)

        # Extract observables from data measurements.
        builder.obs_include([gen.AtLayer(q, "end") for q in self.final_observable_qubits()], obs_index=0)

        if no_mid_reset:
            return builder.circuit
        return builder.circuit.with_inlined_feedback()

    def make_no_reset_ideal_circuit(self) -> stim.Circuit:
        builder = gen.Builder.for_qubits(self.patch.data_set)

        # Prepare individual qubits into the basis needed for the very first interactions.
        x_init_qubits = set()
        if self.basis == 'X':
            x_init_qubits ^= self.patch.data_set
        builder.gate("RX", self.patch.data_set & x_init_qubits)
        builder.gate("R", self.patch.data_set - x_init_qubits)
        builder.tick()

        # Perform initial rounds to get the system into a steady state for the loop.
        if self.start_backwards:
            cmp_b_first = "pre_initB"
            self.make_circuit_layer_type_B_measure(out=builder, cmp_layer=None, save_layer="pre_initB")
            self.make_circuit_layer_type_BtoA(out=builder)
        else:
            cmp_b_first = None
        self.make_circuit_layer_type_A_measure(out=builder, cmp_layer=None, save_layer="initA")
        self.make_circuit_layer_type_AtoB(out=builder)

        self.make_circuit_layer_type_B_measure(out=builder, cmp_layer=cmp_b_first, save_layer="initB")

        return builder.circuit.with_inlined_feedback()
