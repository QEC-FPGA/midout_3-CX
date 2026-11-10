from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Set

import stim


_MEASURE_1Q_NAMES = {'M', 'MZ', 'MX', 'MY', 'MR', 'MRZ', 'MRX', 'MRY'}


@dataclass(frozen=True)
class _DetectorInfo:
    index: int
    t: float
    uses_final_data_measurement: bool


def with_final_data_detectors_on_separate_time_layer(
        circuit: stim.Circuit,
        *,
        final_data_qubits: Iterable[int],
) -> stim.Circuit:
    """Moves final data-readout detectors to a later source-coordinate layer when needed."""
    final_data_qubits = frozenset(final_data_qubits)
    if not final_data_qubits:
        return circuit

    final_data_measurement_indices = _final_data_measurement_indices(
        circuit=circuit,
        final_data_qubits=final_data_qubits,
    )
    detector_infos = _detector_infos(
        circuit=circuit,
        final_data_measurement_indices=final_data_measurement_indices,
    )
    if not detector_infos:
        return with_detectors_sorted_by_time_layer(circuit)

    final_detector_infos = [
        info for info in detector_infos
        if info.uses_final_data_measurement
    ]
    if not final_detector_infos:
        return with_detectors_sorted_by_time_layer(circuit)

    final_detector_ts = {info.t for info in final_detector_infos}
    if len(final_detector_ts) != 1:
        raise NotImplementedError(
            f'Final data-readout detectors are spread across multiple source time layers: {final_detector_ts!r}.'
        )
    final_detector_t, = final_detector_ts
    max_t = max(info.t for info in detector_infos)
    if final_detector_t != max_t:
        raise NotImplementedError(
            f'Final data-readout detectors are at t={final_detector_t!r}, but the final detector layer is t={max_t!r}.'
        )

    max_t_infos = [
        info for info in detector_infos
        if info.t == max_t
    ]
    if all(info.uses_final_data_measurement for info in max_t_infos):
        return with_detectors_sorted_by_time_layer(circuit)

    final_detector_indices = {
        info.index for info in final_detector_infos
    }
    shifted = _with_detectors_shifted_to_next_time_layer(
        circuit=circuit,
        detector_indices=final_detector_indices,
    )
    return with_detectors_sorted_by_time_layer(shifted)


def with_detectors_sorted_by_time_layer(circuit: stim.Circuit) -> stim.Circuit:
    """Stable-sorts each contiguous DETECTOR block by the detector time coordinate."""
    out = stim.Circuit()
    pending_detectors: List[stim.CircuitInstruction] = []

    def flush_detectors() -> None:
        pending_detectors.sort(key=_detector_time_coord)
        for detector in pending_detectors:
            out.append(detector)
        pending_detectors.clear()

    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise NotImplementedError('Detector time-layer sorting does not support REPEAT blocks.')

        if instruction.name == 'DETECTOR':
            pending_detectors.append(instruction)
        else:
            flush_detectors()
            out.append(instruction)

    flush_detectors()
    return out


def _final_data_measurement_indices(
        *,
        circuit: stim.Circuit,
        final_data_qubits: FrozenSet[int],
) -> Set[int]:
    last_measurement_index_by_qubit = {}
    measurement_count = 0

    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise NotImplementedError('Final detector layer separation does not support REPEAT blocks.')

        if instruction.name in _MEASURE_1Q_NAMES:
            for target in instruction.targets_copy():
                if not target.is_qubit_target:
                    raise NotImplementedError(f'Unsupported measurement target in {instruction!r}.')
                if target.value in final_data_qubits:
                    last_measurement_index_by_qubit[target.value] = measurement_count
                measurement_count += 1
        elif instruction.name == 'MPP':
            for _ in _iter_mpp_products(instruction.targets_copy()):
                measurement_count += 1

    missing_qubits = final_data_qubits - frozenset(last_measurement_index_by_qubit.keys())
    if missing_qubits:
        raise NotImplementedError(f'Final data qubits were not measured: {sorted(missing_qubits)!r}.')

    return set(last_measurement_index_by_qubit.values())


def _detector_time_coord(instruction: stim.CircuitInstruction) -> float:
    args = instruction.gate_args_copy()
    if len(args) < 3:
        raise NotImplementedError(f'DETECTOR without xyz coordinates cannot be time-layer sorted: {instruction!r}')
    return args[2]


def _detector_infos(
        *,
        circuit: stim.Circuit,
        final_data_measurement_indices: Set[int],
) -> List[_DetectorInfo]:
    detector_infos: List[_DetectorInfo] = []
    measurement_count = 0
    detector_index = 0

    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise NotImplementedError('Final detector layer separation does not support REPEAT blocks.')

        if instruction.name in _MEASURE_1Q_NAMES:
            measurement_count += len(instruction.targets_copy())
        elif instruction.name == 'MPP':
            for _ in _iter_mpp_products(instruction.targets_copy()):
                measurement_count += 1
        elif instruction.name == 'DETECTOR':
            args = instruction.gate_args_copy()
            if len(args) < 3:
                raise NotImplementedError(f'Final data-readout detector without xyz coordinates: {instruction!r}')
            used_measurements = {
                measurement_count + target.value
                for target in instruction.targets_copy()
                if target.is_measurement_record_target
            }
            detector_infos.append(_DetectorInfo(
                index=detector_index,
                t=args[2],
                uses_final_data_measurement=bool(used_measurements & final_data_measurement_indices),
            ))
            detector_index += 1

    return detector_infos


def _with_detectors_shifted_to_next_time_layer(
        *,
        circuit: stim.Circuit,
        detector_indices: Set[int],
) -> stim.Circuit:
    out = stim.Circuit()
    detector_index = 0
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            raise NotImplementedError('Final detector layer separation does not support REPEAT blocks.')

        if instruction.name != 'DETECTOR':
            out.append(instruction)
            continue

        if detector_index not in detector_indices:
            out.append(instruction)
            detector_index += 1
            continue

        args = instruction.gate_args_copy()
        if len(args) < 3:
            raise NotImplementedError(f'Final data-readout detector without xyz coordinates: {instruction!r}')
        args[2] += 1
        out.append(instruction.name, instruction.targets_copy(), args)
        detector_index += 1
    return out


def _iter_mpp_products(targets: Iterable[stim.GateTarget]) -> Iterable[None]:
    previous_was_combiner = False
    for target in targets:
        if target.is_combiner:
            previous_was_combiner = True
        else:
            if not previous_was_combiner:
                yield None
            previous_was_combiner = False
