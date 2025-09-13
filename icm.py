#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ICM калькулятор для турниров (Independent Chip Model)
"""

import argparse
import json
import sys
import math
from functools import lru_cache
from typing import List, Tuple, Dict


# ------------------------------
# Парсинг аргументов командной строки
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="ICM калькулятор для покерных турниров"
    )
    parser.add_argument(
        "--stacks",
        nargs="+",
        type=float,
        help="Стеки игроков (например: 250000 180000 120000)"
    )
    parser.add_argument(
        "--payouts",
        nargs="+",
        type=str,
        help="Выплаты за места (например: 5000 3000 2000 или 50% 30% 20%)"
    )
    parser.add_argument(
        "--names",
        nargs="+",
        type=str,
        help="Имена игроков (например: Alice Bob Carol)"
    )
    parser.add_argument(
        "--prize-pool",
        type=float,
        help="Общий призовой фонд (обязательно, если выплаты в процентах)"
    )
    parser.add_argument(
        "--currency",
        type=str,
        default="",
        help="Валюта (например: USD, €, ₽)"
    )
    parser.add_argument(
        "--normalize-payouts",
        action="store_true",
        help="Нормализовать выплаты к призовому фонду"
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Число знаков после запятой"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Вывод в формате JSON"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Только результат без лишнего текста"
    )
    return parser.parse_args()


# ------------------------------
# Валидация и подготовка данных
# ------------------------------
def parse_payouts(raw: List[str], prize_pool: float = None) -> List[float]:
    """Парсим список выплат: проценты или абсолютные значения"""
    payouts = []
    if not raw:
        return payouts
    if any(p.endswith("%") for p in raw):
        if not prize_pool:
            sys.exit("Ошибка: для процентов нужно указать --prize-pool")
        total_percent = 0
        for p in raw:
            if not p.endswith("%"):
                sys.exit("Ошибка: выплаты должны быть либо все в %, либо все в абсолютных значениях")
            val = float(p.strip("%"))
            total_percent += val
            payouts.append(prize_pool * val / 100.0)
        if abs(total_percent - 100) > 1e-6:
            print("⚠️ Предупреждение: сумма процентов не равна 100")
    else:
        payouts = [float(p) for p in raw]
    return payouts


def validate_inputs(stacks: List[float], payouts: List[float]):
    if len(stacks) < 2:
        sys.exit("Ошибка: нужно минимум 2 игрока")
    if len(payouts) < 1:
        sys.exit("Ошибка: нужно указать хотя бы одну выплату")
    if len(payouts) > len(stacks):
        sys.exit("Ошибка: число выплат больше числа игроков")
    if any(s <= 0 for s in stacks):
        sys.exit("Ошибка: все стеки должны быть положительными")
    if any(p < 0 for p in payouts):
        sys.exit("Ошибка: выплаты не могут быть отрицательными")
    for i in range(1, len(payouts)):
        if payouts[i] > payouts[i - 1]:
            print("⚠️ Предупреждение: выплаты возрастают, что необычно для турниров")


# ------------------------------
# Логика ICM (с мемоизацией)
# ------------------------------
def icm_ev(stacks: List[float], payouts: List[float]) -> Tuple[List[float], List[List[float]]]:
    """
    Расчёт ICM EV для игроков
    Возвращает:
      ev[i] - денежное эквити игрока i
      probs[i][k] - вероятность занять k-е место (k=0..M-1)
    """

    N = len(stacks)
    M = len(payouts)
    total_chips = sum(stacks)

    @lru_cache(maxsize=None)
    def place_probs(remaining: Tuple[int], k: int) -> Dict[int, float]:
        """Вероятности занять k-е место для заданного набора игроков"""
        rem = list(remaining)
        sub_stacks = [stacks[i] for i in rem]
        sum_chips = sum(sub_stacks)

        probs = {i: 0.0 for i in rem}

        if k == 1:
            for i in rem:
                probs[i] = stacks[i] / sum_chips
            return probs

        # Рекурсивный шаг
        for j in rem:
            pr_j_first = stacks[j] / sum_chips
            new_remaining = tuple(x for x in rem if x != j)
            sub_probs = place_probs(new_remaining, k - 1)
            for i in sub_probs:
                probs[i] += pr_j_first * sub_probs[i]
        return probs

    # Вычисляем EV и вероятности
    ev = [0.0] * N
    probs_matrix = [[0.0] * M for _ in range(N)]

    full_players = tuple(range(N))
    for k in range(1, M + 1):
        probs_k = place_probs(full_players, k)
        for i in probs_k:
            prob = probs_k[i]
            probs_matrix[i][k - 1] = prob
            ev[i] += prob * payouts[k - 1]

    return ev, probs_matrix


# ------------------------------
# Вывод результатов
# ------------------------------
def format_table(stacks, names, payouts, ev, probs, precision=2, currency=""):
    N = len(stacks)
    total_chips = sum(stacks)

    header = ["#", "Name", "Stack", "%Chips", "ICM EV", "Finish Probabilities"]
    rows = []
    for i in range(N):
        chip_pct = stacks[i] / total_chips * 100
        probs_str = " | ".join(
            [f"{j+1}st: {100*probs[i][j]:.{precision}f}%" for j in range(len(payouts))]
        )
        rows.append([
            i + 1,
            names[i],
            int(stacks[i]),
            f"{chip_pct:.{precision}f}%",
            f"{ev[i]:.{precision}f}{currency}",
            probs_str
        ])

    # Форматированный вывод (без внешних библиотек)
    col_widths = [max(len(str(row[c])) for row in rows + [header]) for c in range(len(header))]
    fmt_row = "  ".join("{:<" + str(w) + "}" for w in col_widths)

    lines = [fmt_row.format(*header)]
    for row in rows:
        lines.append(fmt_row.format(*row))
    return "\n".join(lines)


# ------------------------------
# Основная функция
# ------------------------------
def main():
    args = parse_args()

    stacks = args.stacks
    payouts = parse_payouts(args.payouts, args.prize_pool)
    validate_inputs(stacks, payouts)

    N = len(stacks)
    names = args.names if args.names and len(args.names) == N else [f"P{i+1}" for i in range(N)]

    ev, probs = icm_ev(stacks, payouts)

    if args.json:
        result = {
            "players": [
                {
                    "index": i + 1,
                    "name": names[i],
                    "stack": stacks[i],
                    "chip_share": stacks[i] / sum(stacks),
                    "finish_probabilities": probs[i],
                    "icm_ev": ev[i],
                }
                for i in range(N)
            ],
            "inputs": {"stacks": stacks, "payouts": payouts, "currency": args.currency},
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if not args.quiet:
            print(f"ICM Calculator | Players: {N} | Paid places: {len(payouts)}")
        table = format_table(stacks, names, payouts, ev, probs, args.precision, args.currency)
        print(table)


if __name__ == "__main__":
    main()