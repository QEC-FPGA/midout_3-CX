from typing import Dict, Iterable, List, Tuple

import stim


_MEASURE_BASES = {
    'M': 'Z',
    'MZ': 'Z',
    'MX': 'X',
    'MY': 'Y',
    'MR': 'Z',
    'MRZ': 'Z',
    'MRX': 'X',
    'MRY': 'Y',
}

_MEASURE_WITH_RESET_TO_MEASURE = {
    'MR': 'M',
    'MRZ': 'M',
    'MRX': 'MX',
    'MRY': 'MY',
}

_RESET_NAMES = {'R', 'RZ', 'RX', 'RY'}
_ANNOTATION_NAMES_WITH_RECS = {'DETECTOR', 'OBSERVABLE_INCLUDE'}


class NoResetSyndromeTracker:
    """Tracks no-reset raw measurements and their virtual reset-equivalent syndromes."""

    def __init__(self):
        self.previous_raw_measurement: Dict[Tuple[int, str], int] = {}
        self.virtual_measurement_indices: Dict[int, List[int]] = {}

    def record_1q_measurement(self, *, qubit: int, basis: str, measurement_index: int) -> None:
        previous = self.previous_raw_measurement.get((qubit, basis))
        if previous is None:
            self.virtual_measurement_indices[measurement_index] = [measurement_index]
        else:
            self.virtual_measurement_indices[measurement_index] = [previous, measurement_index]
        self.previous_raw_measurement[(qubit, basis)] = measurement_index

    def record_identity_measurement(self, *, measurement_index: int) -> None:
        self.virtual_measurement_indices[measurement_index] = [measurement_index]

    def mapped_measurement_indices(self, measurement_index: int) -> List[int]:
        return self.virtual_measurement_indices.get(measurement_index, [measurement_index])


def without_mid_circuit_resets_by_detectors(circuit: stim.Circuit) -> stim.Circuit:
    """Removes mid-circuit resets and rewrites detectors using no-reset syndrome rules.

    If raw no-reset measurements are n[g, j], this rewrites references to the
    reset-equivalent syndrome m[g, j] into references to n[g, j - 1] ^ n[g, j].
    The first raw measurement has no previous term.
    """
    tracker = NoResetSyndromeTracker()
    out = stim.Circuit()
    measurement_count = 0
    seen_measurement = False

    for instruction in circuit.flattened():
        name = instruction.name
        targets = instruction.targets_copy()
        args = instruction.gate_args_copy()

        if name in _RESET_NAMES:
            if not seen_measurement:
                out.append(name, targets, args)
            continue

        if name in _MEASURE_BASES:
            seen_measurement = True
            measure_name = _MEASURE_WITH_RESET_TO_MEASURE.get(name, name)
            out.append(measure_name, targets, args)
            basis = _MEASURE_BASES[name]
            for target in targets:
                if not target.is_qubit_target:
                    raise NotImplementedError(f'Unsupported measurement target in {instruction!r}.')
                tracker.record_1q_measurement(
                    qubit=target.value,
                    basis=basis,
                    measurement_index=measurement_count,
                )
                measurement_count += 1
            continue

        if name == 'MPP':
            seen_measurement = True
            out.append(name, targets, args)
            for _ in _iter_mpp_products(targets):
                tracker.record_identity_measurement(measurement_index=measurement_count)
                measurement_count += 1
            continue

        if name in _ANNOTATION_NAMES_WITH_RECS:
            out.append(name, _rewrite_rec_targets(
                targets=targets,
                measurement_count=measurement_count,
                tracker=tracker,
            ), args)
            continue

        out.append(name, targets, args)

    return out


def _rewrite_rec_targets(
        *,
        targets: Iterable[stim.GateTarget],
        measurement_count: int,
        tracker: NoResetSyndromeTracker,
) -> List[stim.GateTarget]:
    kept_targets: List[stim.GateTarget] = []
    raw_measurement_parity = set()
    for target in targets:
        if not target.is_measurement_record_target:
            kept_targets.append(target)
            continue
        measurement_index = measurement_count + target.value
        for raw_index in tracker.mapped_measurement_indices(measurement_index):
            if raw_index in raw_measurement_parity:
                raw_measurement_parity.remove(raw_index)
            else:
                raw_measurement_parity.add(raw_index)
    kept_targets.extend(
        stim.target_rec(raw_index - measurement_count)
        for raw_index in sorted(raw_measurement_parity)
    )
    return kept_targets


def _iter_mpp_products(targets: Iterable[stim.GateTarget]) -> Iterable[None]:
    previous_was_combiner = False
    for target in targets:
        if target.is_combiner:
            previous_was_combiner = True
        else:
            if not previous_was_combiner:
                yield None
            previous_was_combiner = False
