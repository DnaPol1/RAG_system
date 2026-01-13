import matplotlib.pyplot as plt
import seaborn as sns


def plot_hit_rate_curve(hit_rates_dict, title="Hit Rate Curve"):
    """
    Рисует кривую Hit Rate@k
    """
    ks = sorted([int(k.split('@')[1]) for k in hit_rates_dict.keys()])
    values = [hit_rates_dict[f'hit_rate@{k}'] for k in ks]

    plt.figure(figsize=(10, 6))
    plt.plot(ks, values, marker='o', linewidth=2, markersize=8)
    plt.xlabel('k (количество документов)', fontsize=12)
    plt.ylabel('Hit Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(ks)
    plt.ylim(0, 1.05)

    # Добавляем значения на график
    for k, v in zip(ks, values):
        plt.text(k, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    return plt
