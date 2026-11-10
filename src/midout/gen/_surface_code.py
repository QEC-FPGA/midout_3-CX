from midout import gen


def checkerboard_basis(q: complex) -> str:
    """Classifies a coordinate as X type or Z type according to a checkerboard.
    """
    is_x = int(q.real + q.imag) & 1 == 0
    return 'X' if is_x else 'Z'


def surface_code_patch(*, distance: int) -> gen.Patch:
    top_bot_basis = 'Z'
    left_right_basis = 'X'

    ur, ul, dl, dr = [(0.5 + 0.5j)*1j**d for d in range(4)]
    order_z = [ul, ur, dl, dr]
    order_ᴎ = [ul, dl, ur, dr]

    data_qubits = {
        x + 1j*y
        for x in range(distance)
        for y in range(distance)
    }
    potential_measure_qubits = {
        q + d
        for q in data_qubits
        for d in order_z
    }

    tiles = []
    for m in gen.sorted_complex(potential_measure_qubits):
        on_top = m.imag < 0
        on_bottom = m.imag > distance - 1
        on_left = m.real < 0
        on_right = m.real > distance - 1
        tile_basis = checkerboard_basis(m)
        if (on_top or on_bottom) and tile_basis != top_bot_basis:
            continue
        if (on_left or on_right) and tile_basis != left_right_basis:
            continue
        order = order_z if tile_basis == 'Z' else order_ᴎ
        tiles.append(gen.Tile(
            bases=tile_basis,
            measurement_qubit=2*m,
            ordered_data_qubits=[2*m + 2*d if m + d in data_qubits else None for d in order],
        ))

    return gen.Patch(tiles)


def lattice_surgery_surface_code_patch(*, distancex, distancez, l: int) -> gen.Patch:
    top_bot_basis = 'Z'
    left_right_basis = 'X'

    ur, ul, dl, dr = [(0.5 + 0.5j)*1j**d for d in range(4)]
    order_z = [ul, ur, dl, dr]
    order_ᴎ = [ul, dl, ur, dr]

    data_qubits = {
        x + 1j*y
        for x in range(distancex)
        for y in range(2*distancez+l)
    }
    potential_measure_qubits = {
        q + d
        for q in data_qubits
        for d in order_z
    }

    tiles = []
    for m in gen.sorted_complex(potential_measure_qubits):
        on_top = m.imag < 0
        on_bottom = m.imag > 2*distancez+l - 1
        on_left = m.real < 0
        on_right = m.real > distancex - 1
        tile_basis = checkerboard_basis(m)
        if (on_top or on_bottom) and tile_basis != top_bot_basis:
            continue
        if (on_left or on_right) and tile_basis != left_right_basis:
            continue
        order = order_z if tile_basis == 'Z' else order_ᴎ
        tiles.append(gen.Tile(
            bases=tile_basis,
            measurement_qubit=2*m,
            ordered_data_qubits=[2*m + 2*d if m + d in data_qubits else None for d in order],
        ))

    data_qubits0 = {
        x + 1j*y
        for x in range(distancex)
        for y in range(distancez)
    }
    potential_measure_qubits0 = {
        q + d
        for q in data_qubits0
        for d in order_z
    }
    
    tiles0 = []
    for m in gen.sorted_complex(potential_measure_qubits0):
        on_top = m.imag < 0
        on_bottom = m.imag > distancez - 1
        on_left = m.real < 0
        on_right = m.real > distancex - 1
        tile_basis = checkerboard_basis(m)
        if (on_top or on_bottom) and tile_basis != top_bot_basis:
            continue
        if (on_left or on_right) and tile_basis != left_right_basis:
            continue
        order = order_z if tile_basis == 'Z' else order_ᴎ
        tiles0.append(gen.Tile(
            bases=tile_basis,
            measurement_qubit=2*m,
            ordered_data_qubits=[2*m + 2*d if m + d in data_qubits0 else None for d in order],
        ))

    data_qubits1 = {
        x + 1j*y
        for x in range(distancex)
        for y in range(distancez+l, 2*distancez+l)
    }
    potential_measure_qubits1 = {
        q + d
        for q in data_qubits1
        for d in order_z
    }
    
    tiles1 = []
    for m in gen.sorted_complex(potential_measure_qubits1):
        on_top = m.imag < distancez+l
        on_bottom = m.imag > 2*distancez+l - 1
        on_left = m.real < 0
        on_right = m.real > distancex - 1
        tile_basis = checkerboard_basis(m)
        if (on_top or on_bottom) and tile_basis != top_bot_basis:
            continue
        if (on_left or on_right) and tile_basis != left_right_basis:
            continue
        order = order_z if tile_basis == 'Z' else order_ᴎ
        tiles1.append(gen.Tile(
            bases=tile_basis,
            measurement_qubit=2*m,
            ordered_data_qubits=[2*m + 2*d if m + d in data_qubits1 else None for d in order],
        ))

    return gen.Patch(tiles), gen.Patch(tiles0), gen.Patch(tiles1)
