import os
import corner
import matplotlib.pyplot as plt


def save_trace_plots(sampler, param_names, plots_dir):
    """
    Generate and save one trace plot per parameter.
    """
    traceplots_dir = os.path.join(plots_dir, "traceplots")
    os.makedirs(traceplots_dir, exist_ok=True)

    samples = sampler.get_chain()

    for i, pname in enumerate(param_names):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(samples[:, :, i], alpha=0.3)
        ax.set_ylabel(pname)
        ax.set_xlabel("Step")
        ax.set_title(f"Trace plot: {pname}")
        fig.tight_layout()

        fname = os.path.join(traceplots_dir, f"trace_{pname}.png")
        fig.savefig(fname, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

        print(f"Saved trace plot for '{pname}' in '{fname}'")


def save_corner_plot(
    flat_samples,
    param_names,
    usar_armonicos_lentos,
    slow_harmonics,
    plots_dir,
    polynomial_degree=2,
    mode="reduced",
):
    """
    Generate and save the corner plot.

    Parameters
    ----------
    polynomial_degree : int, optional
        Degree of the polynomial trend. Allowed values are 1, 2, and 3.
    mode : str
        - "reduced": polynomial coefficients + seasonal-cycle parameters
        - "full": all parameters
    """

    if polynomial_degree not in [1, 2, 3]:
        raise ValueError("polynomial_degree must be 1, 2, or 3")

    if mode == "full":
        corner_indices = list(range(flat_samples.shape[1]))
        flat_subset = flat_samples[:, corner_indices]
        corner_labels = param_names

        output_name = "corner_plot_full.png"

    elif mode == "reduced":
        # Polynomial coefficients: a0, a1, ..., aN
        n_poly_params = polynomial_degree + 1
        corner_indices = list(range(n_poly_params))

        # Offset produced by the slow harmonics
        despl = 2 * len(slow_harmonics) if (usar_armonicos_lentos and len(slow_harmonics) > 0) else 0
        base = n_poly_params + despl

        # b1, c1, bp1, cp1, b2, c2, b3, c3, b4, c4
        corner_indices += list(range(base, base + 10))

        flat_subset = flat_samples[:, corner_indices]

        polynomial_labels = [rf"$a_{i}$" for i in range(n_poly_params)]

        corner_labels = polynomial_labels + [
            r"$b_1$", r"$c_1$", r"$b_1'$", r"$c_1'$",
            r"$b_2$", r"$c_2$", r"$b_3$", r"$c_3$", r"$b_4$", r"$c_4$"
        ]

        output_name = "corner_plot_reduced.png"

    else:
        raise ValueError("mode must be 'reduced' or 'full'")

    fig = corner.corner(
        flat_subset,
        labels=corner_labels,
        show_titles=True,
        quantiles=(0.16, 0.5, 0.84),
        title_fmt=".4f",
        title_kwargs={"fontsize": 16},
        label_kwargs={"fontsize": 30},
        labelpad=0.28,
        max_n_ticks=3,
        use_math_text=True,
    )

    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=14, direction="in", top=True, right=True, pad=2)

    output_path = os.path.join(plots_dir, output_name)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    print(f"Saved corner plot in '{output_path}'")
