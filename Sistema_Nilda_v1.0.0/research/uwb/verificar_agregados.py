"""Recalcula o erro entre sensores a partir das médias de distância arquivadas."""

import csv
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path


CSV_PATH = Path(__file__).with_name("medicoes_agregadas_uwb.csv")
COLUMNS = (
    "distancia_esperada_cm",
    "distancia_detectada_media_cm",
    "erro_absoluto_cm",
    "repeticoes_declaradas",
)


def main():
    distances = set()
    errors = []
    with CSV_PATH.open(encoding="utf-8", newline="") as csv_file:
        rows = csv.DictReader(csv_file)
        if tuple(rows.fieldnames or ()) != COLUMNS:
            raise ValueError("O cabeçalho do CSV não corresponde ao formato arquivado.")
        for row in rows:
            expected, detected, stored_error = (Decimal(row[key]) for key in COLUMNS[:3])
            if not all(value.is_finite() for value in (expected, detected, stored_error)):
                raise ValueError(f"Linha {rows.line_num} com valor não finito.")
            absolute_error = abs(detected - expected)
            if absolute_error != stored_error:
                raise ValueError(f"Erro absoluto inconsistente na linha {rows.line_num}.")
            if row["repeticoes_declaradas"] != "15":
                raise ValueError(f"Número de repetições incompatível na linha {rows.line_num}.")
            distances.add(expected)
            errors.append(absolute_error)

    if len(errors) != 13 or len(distances) != 13 or min(distances) != 25 or max(distances) != 450:
        raise ValueError("Esperadas 13 distâncias distintas entre 25 cm e 450 cm.")

    total = sum(errors)
    print(json.dumps({
        "distancias_agregadas": len(distances),
        "repeticoes_declaradas_por_distancia": 15,
        "soma_erros_absolutos_cm": str(total),
        "mae_agregado_cm": f"{total / len(errors):.2f}",
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (InvalidOperation, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"Falha ao verificar os dados UWB: {error}") from error
