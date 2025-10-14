def reconstruct_keyboard_grid(key_boxes, y_threshold=40):
    """
    Organise les touches détectées en lignes et colonnes pour reconstruire le clavier.
    """
    # Calculer le centre vertical de chaque rectangle
    sorted_boxes = sorted(key_boxes, key=lambda b: (b[0] + b[2]) / 2)

    lines = []
    current_line = []

    for box in sorted_boxes:
        minr, minc, maxr, maxc = box
        center_y = (minr + maxr) / 2
        if not current_line:
            current_line.append(box)
            current_center = center_y
        else:
            if abs(center_y - current_center) <= y_threshold:
                current_line.append(box)
            else:
                current_line.sort(key=lambda b: b[1])  # tri par x
                lines.append(current_line)
                current_line = [box]
                current_center = center_y

    if current_line:
        current_line.sort(key=lambda b: b[1])
        lines.append(current_line)

    return lines


def grid_to_symbols(keyboard_grid, symbols_matrix=None):
    """
    Associe les touches détectées à des symboles.
    """
    grid_with_symbols = []

    for i, line in enumerate(keyboard_grid):
        if symbols_matrix is not None and i < len(symbols_matrix):
            line_symbols = symbols_matrix[i]
            # Associer chaque rectangle à un symbole
            grid_with_symbols.append(
                [
                    line_symbols[j] if j < len(line_symbols) else None
                    for j in range(len(line))
                ]
            )
        else:
            # Pas de symboles fournis → remplir avec None
            grid_with_symbols.append([None] * len(line))

    return grid_with_symbols
