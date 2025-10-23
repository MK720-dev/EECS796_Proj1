#  _*_ coding: utf-8 _*_
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------------------------------------------------
# Load and preprocess data
# --------------------------------------------------------------------
csv_path = r"C:\Users\kchao\OneDrive\Documents\Dossier_Malek\KU\KU 4th Year Courses\EECS 796\Project1\results\optimizer_comparison_summary.csv"
df = pd.read_csv(csv_path)

# Ensure proper string types
df["Optimizer"] = df["Optimizer"].astype(str)
df["Dataset"] = df["Dataset"].astype(str)
df["Architecture"] = df["Architecture"].astype(str)

# Separate custom vs TF
df_custom = df[df["Optimizer"].str.contains("(Custom)", regex=False)].copy()
df_tf = df[df["Optimizer"].str.contains("(TF)", regex=False)].copy()

# ================================
# Rename gradient descent variants
# ================================
def rename_gd_variant(row):
    """Differentiate GD variants based on batch size."""
    opt = str(row["Optimizer"]).strip().lower()
    bs = str(row["BatchSize"]).strip().lower()

    if "gradientdescent" in opt:
        if bs in ["none", "full", "nan"]:
            return "GradientDescent_Full (TF)"
        elif bs in ["1", "1.0"]:
            return "SGD (TF)"
        elif bs in ["20", "20.0"]:
            return "GradientDescent_MiniBatch (TF)"
        else:
            return f"GradientDescent_MiniBatch_{bs} (TF)"
    else:
        return row["Optimizer"]

df_tf["Optimizer"] = df_tf.apply(rename_gd_variant, axis=1)

# Clean optimizer names (remove the suffixes)
df_custom["OptBase"] = df_custom["Optimizer"].str.replace(" (Custom)", "", regex=False)
df_tf["OptBase"] = df_tf["Optimizer"].str.replace(" (TF)", "", regex=False)

df_custom["OptBase"] = df_custom["OptBase"].str.lower()
df_tf["OptBase"] = df_tf["OptBase"].str.lower()

# Define optimizer families
traditional = {"gradientdescent_full", "gradientdescent_minibatch", "sgd", "bfgs"}
adaptive = {"adam", "rmsprop", "adagrad"}

for frame in [df_custom, df_tf]:
    frame["Family"] = frame["OptBase"].apply(
        lambda x: "Traditional" if x in traditional else ("Adaptive" if x in adaptive else "Other")
    )

# --------------------------------------------------------------------
# Create output directories
# --------------------------------------------------------------------
os.makedirs("results/plots/custom_bar_plots", exist_ok=True)
os.makedirs("results/plots/tf_bar_plots", exist_ok=True)

# --------------------------------------------------------------------
# Plot helper function
# --------------------------------------------------------------------
def plot_dual_panel(data, dataset, arch, source_label, save_dir):
    """Generate dual-panel plot for given optimizer subset."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # --- Left Panel: Validation Loss ---
    sns.barplot(
        data=data,
        x="OptBase", y="AvgValLoss", hue="Family",
        palette={"Traditional": "#4C72B0", "Adaptive": "#55A868", "Other": "#C44E52"},
        ax=axes[0], ci=None
    )
    axes[0].errorbar(
        x=range(len(data)),
        y=data["AvgValLoss"],
        yerr=data["StdValLoss"],
        fmt='none', c='black', capsize=4
    )
    axes[0].set_title("Validation Loss", fontsize=12, pad=10)
    axes[0].set_xlabel("Optimizer")
    axes[0].set_ylabel("Avg Val Loss")
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(title="Family", fontsize=9)

    # --- Right Panel: Validation Metric ---
    if data["ValMetric"].notna().any():
        sns.barplot(
            data=data,
            x="OptBase", y="ValMetric", hue="Family",
            palette={"Traditional": "#4C72B0", "Adaptive": "#55A868", "Other": "#C44E52"},
            ax=axes[1], ci=None
        )
        axes[1].set_title("Validation Metric (Accuracy or MSE)", fontsize=12, pad=10)
        axes[1].set_xlabel("Optimizer")
        axes[1].set_ylabel("Metric Value")
        axes[1].grid(axis="y", linestyle="--", alpha=0.6)
        axes[1].tick_params(axis="x", rotation=25)
        axes[1].legend_.remove()
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No metric available", ha="center", va="center", fontsize=11, alpha=0.6)

    plt.suptitle(f"{dataset} - {arch} Network ({source_label})", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout(pad=2.0)

    save_path = os.path.join(save_dir, f"{dataset}_{arch}_{source_label}_dual_comparison.png".replace(" ", "_"))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# --------------------------------------------------------------------
# Generate plots for both sources
# --------------------------------------------------------------------
for (dataset, arch), sub in df_custom.groupby(["Dataset", "Architecture"]):
    plot_dual_panel(sub, dataset, arch, "Custom", "results/plots/custom_bar_plots")

for (dataset, arch), sub in df_tf.groupby(["Dataset", "Architecture"]):
    plot_dual_panel(sub, dataset, arch, "TF", "results/plots/tf_bar_plots")

print("Dual-panel comparison plots generated for both Custom and TF optimizers.")
print("Saved under: results/plots/custom_bar_plots and results/plots/tf_bar_plots")


