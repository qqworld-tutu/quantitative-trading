def momentum_scores(price_map, idx, window):
    scores = {}
    for name, prices in price_map.items():
        scores[name] = prices[idx] / prices[idx - window] - 1
    return scores


def top_n_mask(scores, top_n):
    chosen = {name: 0 for name in scores}
    for name, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]:
        chosen[name] = 1
    return chosen


def run_rule_strategy(price_map, window, top_n, initial_capital):
    names = list(price_map)
    n = len(next(iter(price_map.values())))
    values = [initial_capital]
    holdings = []
    for idx in range(1, n):
        if idx < window:
            holdings.append({name: 1 / len(names) for name in names})
            gross = sum(price_map[name][idx] / price_map[name][idx - 1] for name in names) / len(names)
        else:
            mask = top_n_mask(momentum_scores(price_map, idx - 1, window), top_n)
            selected = [name for name, flag in mask.items() if flag]
            holdings.append({name: (1 / len(selected) if name in selected else 0.0) for name in names})
            gross = sum(price_map[name][idx] / price_map[name][idx - 1] for name in selected) / len(selected)
        values.append(values[-1] * gross)
    return values, holdings

