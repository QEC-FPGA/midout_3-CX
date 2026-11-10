from typing import Dict, Iterable, Optional

import stim

from midout.gen._patch import Patch


def with_only_detectors_from_basis(
        circuit: stim.Circuit,
        *,
        patches: Iterable[Patch],
        detector_basis: Optional[str],
) -> stim.Circuit:
    if detector_basis is None:
        return circuit
    if detector_basis not in ['X', 'Z']:
        raise ValueError(f"{detector_basis=}")

    patches = tuple(patches)
    pos_to_basis = _detector_pos_to_basis(patches)
    out = stim.Circuit()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(
                repeat_count=instruction.repeat_count,
                body=with_only_detectors_from_basis(
                    instruction.body_copy(),
                    patches=patches,
                    detector_basis=detector_basis,
                ),
            ))
            continue

        if instruction.name == 'DETECTOR':
            basis = _basis_for_detector_instruction(
                instruction=instruction,
                pos_to_basis=pos_to_basis,
            )
            if basis != detector_basis:
                continue

        out.append(instruction)
    return out


def _detector_pos_to_basis(patches: Iterable[Patch]) -> Dict[complex, str]:
    result: Dict[complex, str] = {}
    for patch in patches:
        for tile in patch.tiles:
            if tile.basis is None:
                continue
            pos = tile.measurement_qubit
            old = result.get(pos)
            if old is not None and old != tile.basis:
                raise NotImplementedError(
                    f'Conflicting detector bases at {pos!r}: {old!r} vs {tile.basis!r}.'
                )
            result[pos] = tile.basis
    return result


def _basis_for_detector_instruction(
        *,
        instruction: stim.CircuitInstruction,
        pos_to_basis: Dict[complex, str],
) -> str:
    args = instruction.gate_args_copy()
    if len(args) < 2:
        raise NotImplementedError(f'DETECTOR without xy coordinates cannot be basis-filtered: {instruction!r}')
    pos = args[1] + 1j * args[0]
    basis = pos_to_basis.get(pos)
    if basis is None:
        raise NotImplementedError(f'DETECTOR at {pos!r} does not match any patch tile.')
    return basis
