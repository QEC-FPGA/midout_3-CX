import dataclasses
from typing import List, FrozenSet, Optional, Tuple

import stim

from midout import gen


@dataclasses.dataclass(frozen=True)
class CircuitCase:
    circuit: stim.Circuit
    patches: List[gen.Patch]
    expected_interactions: FrozenSet[str]
    round_end_data_qubit_sets: Tuple[Optional[FrozenSet[int]], ...] = ()
    show_patch_order: bool = False
    show_patch_measure_qubits: bool = False
