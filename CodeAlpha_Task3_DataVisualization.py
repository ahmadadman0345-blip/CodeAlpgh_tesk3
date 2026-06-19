# ============================================================
#   CodeAlpha Internship — Task 3: Data Visualization
#   Dataset  : Titanic (loaded via Seaborn)
#   Charts   : 12 visualizations in one dashboard
#   Author   : [Your Name]
#   LinkedIn : [Your LinkedIn]
#   GitHub   : github.com/[username]/CodeAlpha_DataAnalytics
# ============================================================

# ── STEP 1: Import Libraries ─────────────────────────────────
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

print("✅ Libraries imported successfully!")

# ── STEP 2: Load Dataset ─────────────────────────────────────
df = sns.load_dataset("titanic")

print(f"\n📦 Dataset loaded!")
print(f"   Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📋 Columns:\n   {list(df.columns)}")
print(f"\n🔢 First 5 rows:")
print(df.head())

# ── STEP 3: Basic Info ───────────────────────────────────────
print("\n\n📊 Dataset Info:")
print(df.dtypes)

print("\n\n❌ Missing Values:")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n\n📈 Statistical Summary:")
print(df[["age", "fare", "sibsp", "parch"]].describe().round(2))

# ── STEP 4: Feature Engineering ──────────────────────────────
# Age Groups
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child\n(0–12)", "Teen\n(13–18)", "Young Adult\n(19–35)",
            "Middle Age\n(36–60)", "Senior\n(60+)"]
)

# Family Size
df["family_size"] = df["sibsp"] + df["parch"] + 1

print("\n✅ New features created: age_group, family_size")

# ── STEP 5: Quick Stats Before Plotting ──────────────────────
print("\n\n🎯 Key Statistics:")
print(f"   Overall Survival Rate : {df['survived'].mean()*100:.2f}%")
print(f"   Total Passengers      : {len(df)}")
print(f"   Survived              : {df['survived'].sum()}")
print(f"   Did Not Survive       : {(df['survived']==0).sum()}")

print("\n   Survival by Sex:")
print(df.groupby("sex")["survived"].mean().mul(100).round(2))

print("\n   Survival by Class:")
print(df.groupby("pclass")["survived"].mean().mul(100).round(2))

# ── STEP 6: Color Palette & Style ────────────────────────────
SURVIVE_COLORS = ["#E74C3C", "#2ECC71"]   # Red = No, Green = Yes
CLASS_COLORS   = ["#F1C40F", "#3498DB", "#95A5A6"]
BG_COLOR       = "#F8F9FA"
DARK           = "#1A1A2E"

plt.rcParams.update({
    "font.family"        : "DejaVu Sans",
    "axes.facecolor"     : BG_COLOR,
    "figure.facecolor"   : "white",
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "axes.labelcolor"    : DARK,
    "xtick.color"        : DARK,
    "ytick.color"        : DARK,
    "axes.titleweight"   : "bold",
    "axes.titlesize"     : 12,
    "axes.titlepad"      : 10,
})

print("\n✅ Style configured!")

# ── STEP 7: Create the Figure (4 rows × 3 columns) ───────────
fig = plt.figure(figsize=(24, 28), facecolor="white")

fig.suptitle(
    "🚢  Titanic Survival Analysis  —  Complete Data Visualization Dashboard",
    fontsize=22, fontweight="bold", color=DARK, y=0.98
)

gs = gridspec.GridSpec(
    4, 3, figure=fig,
    hspace=0.50, wspace=0.38,
    top=0.94, bottom=0.04,
    left=0.06, right=0.97
)

axes = [fig.add_subplot(gs[r, c]) for r in range(4) for c in range(3)]

print("\n✅ Figure & GridSpec created (4×3 = 12 charts)")


# ════════════════════════════════════════════════════════════
#   CHART 1 — Overall Survival (Donut / Pie Chart)
# ════════════════════════════════════════════════════════════
ax = axes[0]

counts      = df["survived"].value_counts().sort_index()
labels      = ["Did Not Survive", "Survived"]
wedge_props = dict(width=0.42, edgecolor="white", linewidth=2.5)

wedges, texts, autotexts = ax.pie(
    counts,
    labels       = labels,
    colors       = SURVIVE_COLORS,
    autopct      = "%1.1f%%",
    startangle   = 90,
    wedgeprops   = wedge_props,
    textprops    = {"fontsize": 9, "color": DARK},
    pctdistance  = 0.75
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight("bold")
    at.set_color("white")

ax.set_title("Chart 1 — Overall Survival (Donut Chart)")
ax.annotate(f"Total\n{len(df)}", xy=(0, 0),
            ha="center", va="center", fontsize=11,
            fontweight="bold", color=DARK)

print("   ✅ Chart 1 — Donut Chart done")


# ════════════════════════════════════════════════════════════
#   CHART 2 — Survival Count by Sex (Grouped Bar)
# ════════════════════════════════════════════════════════════
ax = axes[1]

sex_data = df.groupby(["sex", "survived"]).size().unstack()
sex_data.columns = ["Did Not Survive", "Survived"]
sex_data.plot(kind="bar", ax=ax, color=SURVIVE_COLORS,
              edgecolor="white", linewidth=0.8, width=0.65)

ax.set_title("Chart 2 — Survival Count by Sex")
ax.set_xlabel("Sex", fontsize=10)
ax.set_ylabel("Number of Passengers", fontsize=10)
ax.set_xticklabels(["Female", "Male"], rotation=0, fontsize=10)
ax.legend(fontsize=8, framealpha=0.5)

for bar in ax.patches:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 3,
        str(int(bar.get_height())),
        ha="center", va="bottom", fontsize=9, color=DARK
    )

print("   ✅ Chart 2 — Grouped Bar (Sex) done")


# ════════════════════════════════════════════════════════════
#   CHART 3 — Survival Rate by Passenger Class (Bar Chart)
# ════════════════════════════════════════════════════════════
ax = axes[2]

pclass_rate = df.groupby("pclass")["survived"].mean() * 100
bar_colors  = ["#F1C40F", "#3498DB", "#95A5A6"]
bars = ax.bar(
    ["1st Class", "2nd Class", "3rd Class"],
    pclass_rate.values,
    color=bar_colors, edgecolor="white", linewidth=0.8,
    width=0.55
)

ax.set_title("Chart 3 — Survival Rate by Passenger Class")
ax.set_ylabel("Survival Rate (%)", fontsize=10)
ax.set_ylim(0, 80)

for bar, val in zip(bars, pclass_rate.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{val:.1f}%",
        ha="center", fontsize=10, fontweight="bold", color=DARK
    )

print("   ✅ Chart 3 — Bar Chart (Pclass) done")


# ════════════════════════════════════════════════════════════
#   CHART 4 — Age Distribution by Survival (KDE Curve)
# ════════════════════════════════════════════════════════════
ax = axes[3]

for val, label, color in zip([0, 1], ["Did Not Survive", "Survived"], SURVIVE_COLORS):
    sns.kdeplot(
        df[df["survived"] == val]["age"].dropna(),
        ax=ax, fill=True, alpha=0.45,
        color=color, label=label, linewidth=2
    )

ax.set_title("Chart 4 — Age Distribution by Survival (KDE)")
ax.set_xlabel("Age", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_xlim(0, 85)
ax.legend(fontsize=8)
ax.axvline(df[df["survived"]==1]["age"].mean(), color="#2ECC71",
           linestyle="--", alpha=0.7, linewidth=1.5,
           label=f"Survivor avg: {df[df['survived']==1]['age'].mean():.0f} yrs")

print("   ✅ Chart 4 — KDE Age Distribution done")


# ════════════════════════════════════════════════════════════
#   CHART 5 — Fare Distribution by Survival (KDE Log Scale)
# ════════════════════════════════════════════════════════════
ax = axes[4]

for val, label, color in zip([0, 1], ["Did Not Survive", "Survived"], SURVIVE_COLORS):
    data = df[df["survived"] == val]["fare"].dropna()
    data = data[data > 0]
    sns.kdeplot(
        np.log1p(data), ax=ax, fill=True,
        alpha=0.45, color=color, label=label, linewidth=2
    )

ax.set_title("Chart 5 — Fare Distribution by Survival (Log Scale)")
ax.set_xlabel("log(Fare + 1)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.legend(fontsize=8)

# Add annotation
surv_avg   = df[df["survived"]==1]["fare"].mean()
nosurv_avg = df[df["survived"]==0]["fare"].mean()
ax.text(0.97, 0.95, f"Survivor avg fare: ${surv_avg:.0f}\nNon-survivor avg: ${nosurv_avg:.0f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color=DARK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

print("   ✅ Chart 5 — KDE Fare Distribution done")


# ════════════════════════════════════════════════════════════
#   CHART 6 — Survival Rate by Age Group (Bar Chart)
# ════════════════════════════════════════════════════════════
ax = axes[5]

age_rate   = df.groupby("age_group", observed=True)["survived"].mean() * 100
max_val    = age_rate.max()
bar_colors = ["#2ECC71" if v == max_val else "#AED6F1" for v in age_rate.values]

bars = ax.bar(age_rate.index, age_rate.values,
              color=bar_colors, edgecolor="white", linewidth=0.8, width=0.55)

ax.set_title("Chart 6 — Survival Rate by Age Group")
ax.set_ylabel("Survival Rate (%)", fontsize=10)
ax.set_ylim(0, 75)

for bar, val in zip(bars, age_rate.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{val:.1f}%",
        ha="center", fontsize=9, fontweight="bold", color=DARK
    )

ax.text(0.5, -0.18, "★ Children had the highest survival rate",
        transform=ax.transAxes, ha="center", fontsize=8,
        color="#27AE60", style="italic")

print("   ✅ Chart 6 — Bar Chart (Age Group) done")


# ════════════════════════════════════════════════════════════
#   CHART 7 — Heatmap: Sex × Pclass Survival Rate
# ════════════════════════════════════════════════════════════
ax = axes[6]

pivot = df.pivot_table("survived", index="sex", columns="pclass", aggfunc="mean") * 100

sns.heatmap(
    pivot,
    annot=True, fmt=".1f", cmap="RdYlGn", ax=ax,
    linewidths=0.8, linecolor="white",
    cbar_kws={"label": "Survival %", "shrink": 0.85},
    annot_kws={"size": 13, "weight": "bold"}
)

ax.set_title("Chart 7 — Heatmap: Survival Rate % (Sex × Class)")
ax.set_xlabel("Passenger Class", fontsize=10)
ax.set_ylabel("Sex", fontsize=10)
ax.set_xticklabels(["1st", "2nd", "3rd"], rotation=0)

print("   ✅ Chart 7 — Heatmap done")


# ════════════════════════════════════════════════════════════
#   CHART 8 — Correlation Heatmap
# ════════════════════════════════════════════════════════════
ax = axes[7]

corr_df = df[["survived", "pclass", "age", "sibsp", "parch", "fare"]].dropna().corr()
mask    = np.triu(np.ones_like(corr_df, dtype=bool))

sns.heatmap(
    corr_df,
    annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
    mask=mask, linewidths=0.5, linecolor="white",
    cbar_kws={"shrink": 0.85},
    annot_kws={"size": 9}
)

ax.set_title("Chart 8 — Correlation Matrix Heatmap")

print("   ✅ Chart 8 — Correlation Heatmap done")


# ════════════════════════════════════════════════════════════
#   CHART 9 — Scatter Plot: Fare vs Age (by Class)
# ════════════════════════════════════════════════════════════
ax = axes[8]

scatter_colors = {1: "#F1C40F", 2: "#3498DB", 3: "#95A5A6"}
scatter_labels = {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}

for cls in [1, 2, 3]:
    sub = df[df["pclass"] == cls]
    ax.scatter(
        sub["age"], sub["fare"],
        c=scatter_colors[cls], alpha=0.50, s=22,
        label=scatter_labels[cls], edgecolors="none"
    )

ax.set_title("Chart 9 — Fare vs Age (by Passenger Class)")
ax.set_xlabel("Age", fontsize=10)
ax.set_ylabel("Fare ($)", fontsize=10)
ax.set_ylim(-5, 530)
ax.legend(fontsize=8, loc="upper right")

print("   ✅ Chart 9 — Scatter Plot done")


# ════════════════════════════════════════════════════════════
#   CHART 10 — Survival Rate by Embarkation Town (H-Bar)
# ════════════════════════════════════════════════════════════
ax = axes[9]

emb_rate   = df.groupby("embark_town")["survived"].mean() * 100
bar_colors = ["#3498DB", "#E67E22", "#2ECC71"]

bars = ax.barh(
    emb_rate.index, emb_rate.values,
    color=bar_colors, edgecolor="white", height=0.5
)

ax.set_title("Chart 10 — Survival Rate by Embarkation Town")
ax.set_xlabel("Survival Rate (%)", fontsize=10)
ax.set_xlim(0, 72)

for bar, val in zip(bars, emb_rate.values):
    ax.text(
        val + 0.8,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}%",
        va="center", fontsize=10, fontweight="bold", color=DARK
    )

print("   ✅ Chart 10 — Horizontal Bar (Embarkation) done")


# ════════════════════════════════════════════════════════════
#   CHART 11 — Survival Rate by Family Size (Line Chart)
# ════════════════════════════════════════════════════════════
ax = axes[10]

fam_rate = df.groupby("family_size")["survived"].mean() * 100

ax.plot(
    fam_rate.index, fam_rate.values,
    "o-", color="#8E44AD", linewidth=2.5,
    markersize=8, markerfacecolor="white",
    markeredgecolor="#8E44AD", markeredgewidth=2.5
)
ax.fill_between(fam_rate.index, fam_rate.values, alpha=0.12, color="#8E44AD")

# Annotate peak
peak_x = fam_rate.idxmax()
peak_y = fam_rate.max()
ax.annotate(
    f"Peak: {peak_y:.0f}%\n(Family of {peak_x})",
    xy=(peak_x, peak_y),
    xytext=(peak_x + 0.8, peak_y - 12),
    fontsize=8, color="#8E44AD",
    arrowprops=dict(arrowstyle="->", color="#8E44AD", lw=1.2)
)

ax.set_title("Chart 11 — Survival Rate by Family Size (Line)")
ax.set_xlabel("Family Size  (Self + Siblings + Parents/Children)", fontsize=9)
ax.set_ylabel("Survival Rate (%)", fontsize=10)
ax.set_xticks(fam_rate.index)
ax.set_ylim(0, 90)

print("   ✅ Chart 11 — Line Chart (Family Size) done")


# ════════════════════════════════════════════════════════════
#   CHART 12 — Box Plot: Fare by Class & Survival
# ════════════════════════════════════════════════════════════
ax = axes[11]

sns.boxplot(
    data=df, x="pclass", y="fare", hue="survived",
    palette=SURVIVE_COLORS, ax=ax,
    width=0.6, fliersize=3, linewidth=1,
    medianprops={"linewidth": 2.5, "color": "white"}
)

ax.set_title("Chart 12 — Fare Distribution by Class & Survival (Boxplot)")
ax.set_xlabel("Passenger Class", fontsize=10)
ax.set_ylabel("Fare ($)", fontsize=10)
ax.set_xticklabels(["1st Class", "2nd Class", "3rd Class"])
ax.set_ylim(-5, 330)

handles = [
    mpatches.Patch(color=c, label=l)
    for c, l in zip(SURVIVE_COLORS, ["Did Not Survive", "Survived"])
]
ax.legend(handles=handles, fontsize=8)

print("   ✅ Chart 12 — Boxplot (Fare by Class) done")


# ── STEP 8: Save the Dashboard ───────────────────────────────
output_path = "Titanic_Visualization_Dashboard.png"
plt.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
print(f"\n\n💾 Dashboard saved as → {output_path}")


# ── STEP 9: Show the Dashboard ───────────────────────────────
plt.show()
print("\n🎉 Task 3 Complete — All 12 charts displayed!")


# ── STEP 10: Print Final Summary ─────────────────────────────
print("\n" + "="*60)
print("   SUMMARY OF KEY INSIGHTS")
print("="*60)

survival_rate = df["survived"].mean() * 100
female_rate   = df[df["sex"]=="female"]["survived"].mean() * 100
male_rate     = df[df["sex"]=="male"]["survived"].mean() * 100
class1_rate   = df[df["pclass"]==1]["survived"].mean() * 100
class3_rate   = df[df["pclass"]==3]["survived"].mean() * 100
child_rate    = df[df["age"] <= 12]["survived"].mean() * 100

print(f"\n  Overall Survival Rate  : {survival_rate:.1f}%")
print(f"  Female Survival Rate   : {female_rate:.1f}%")
print(f"  Male Survival Rate     : {male_rate:.1f}%")
print(f"  1st Class Survival     : {class1_rate:.1f}%")
print(f"  3rd Class Survival     : {class3_rate:.1f}%")
print(f"  Children Survival      : {child_rate:.1f}%")
print(f"\n  Charts Created         : 12")
print(f"  Chart Types Used       : Donut, Grouped Bar, KDE,")
print(f"                           Heatmap, Scatter, H-Bar,")
print(f"                           Line Chart, Boxplot")
print("\n" + "="*60)
print("   CodeAlpha Internship — Task 3 Completed ✅")
print("="*60)
