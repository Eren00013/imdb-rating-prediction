from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET = "imdb_score"

NAVY = "#102A56"
BLUE = "#1B67A5"
ORANGE = "#E36A2E"
TEXT = "#202633"
MUTED = "#566273"
BORDER = "#F0A06D"
GRID = "#D7DEE8"
TEXT_LINE_SPACING_SCALE = 1.1


def find_data_path():
    candidates = [
        SCRIPT_DIR.parent / "data" / "movie_metadata.csv",
        SCRIPT_DIR / "movie_metadata.csv",
        SCRIPT_DIR.parent / "_py" / "movie_metadata.csv",
        SCRIPT_DIR.parent / "movie_metadata.csv",
        SCRIPT_DIR.parent.parent / "_py" / "movie_metadata.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("movie_metadata.csv was not found.")


def output_directory():
    return SCRIPT_DIR.parent / "poster"


def register_fonts():
    font_dir = SCRIPT_DIR / "fonts"
    mapping = {
        "regular": "WorkSans-Regular.ttf",
        "bold": "WorkSans-Bold.ttf",
        "italic": "WorkSans-Italic.ttf",
    }
    fonts = {}
    for key, filename in mapping.items():
        path = font_dir / filename
        if path.exists():
            fm.fontManager.addfont(str(path))
            fonts[key] = fm.FontProperties(fname=str(path))
    fonts.setdefault("regular", fm.FontProperties(family="DejaVu Sans"))
    fonts.setdefault("bold", fm.FontProperties(family="DejaVu Sans", weight="bold"))
    fonts.setdefault("italic", fm.FontProperties(family="DejaVu Sans", style="italic"))
    return fonts


FONTS = register_fonts()


def de(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


def add_card(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.005, 0.005),
            0.99,
            0.99,
            boxstyle="round,pad=0.004,rounding_size=0.018",
            facecolor="white",
            edgecolor=BORDER,
            linewidth=1.3,
            transform=ax.transAxes,
        )
    )
    return ax


def heading(ax, number, title, size=22, x=0.035, y=0.91):
    if number:
        ax.text(
            x,
            y,
            number,
            fontproperties=FONTS["bold"],
            fontsize=size + 3,
            color=ORANGE,
            va="top",
        )
        x += 0.048
    ax.text(
        x, y, title, fontproperties=FONTS["bold"], fontsize=size, color=NAVY, va="top"
    )


def body(ax, text, x, y, size, spacing=1.3):
    ax.text(
        x,
        y,
        text,
        fontproperties=FONTS["regular"],
        fontsize=size,
        color=TEXT,
        va="top",
        linespacing=spacing * TEXT_LINE_SPACING_SCALE,
    )


def style_axis(ax, tick_size=13.5):
    ax.set_facecolor("white")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=MUTED, labelsize=tick_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONTS["regular"])


data_path = find_data_path()
out_dir = output_directory()
out_dir.mkdir(parents=True, exist_ok=True)

df_raw = pd.read_csv(data_path)
df = df_raw.drop_duplicates().reset_index(drop=True)
removed_duplicates = len(df_raw) - len(df)
rows, columns = df.shape
numeric_count = len(df.select_dtypes(include=[np.number]).columns)
text_count = columns - numeric_count
stats = df[TARGET].describe()
missing_percent = df.isna().mean().mul(100)

corr = df.select_dtypes(include=[np.number]).corr()
target_corr = corr[TARGET]
correlation_labels = {
    "num_voted_users": "Nutzerstimmen",
    "num_critic_for_reviews": "Kritikerrezensionen",
    "num_user_for_reviews": "Nutzerrezensionen",
    "duration": "Laufzeit",
    "movie_facebook_likes": "Film-Facebook-Likes",
    "cast_total_facebook_likes": "Cast-Facebook-Likes",
    "director_facebook_likes": "Regie-Facebook-Likes",
    "actor_1_facebook_likes": "Hauptdarsteller-Likes",
    "actor_2_facebook_likes": "Nebendarsteller-Likes",
    "actor_3_facebook_likes": "Drittdarsteller-Likes",
    "facenumber_in_poster": "Gesichter im Poster",
    "gross": "Einspielergebnis",
    "budget": "Budget",
    "title_year": "Erscheinungsjahr",
    "aspect_ratio": "Seitenverhältnis",
}
display_correlations = (
    target_corr.drop(index=TARGET).sort_values(key=lambda values: values.abs()).tail(8)
)

model_data = df.select_dtypes(include=[np.number]).dropna()
X = model_data.drop(columns=[TARGET])
y = model_data[TARGET]
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42
)

linear_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]
).fit(X_train, y_train)
random_forest = RandomForestRegressor(n_estimators=100, random_state=42).fit(
    X_train, y_train
)
linear_val_r2 = r2_score(y_val, linear_model.predict(X_val))
forest_val_r2 = r2_score(y_val, random_forest.predict(X_val))
forest_train_r2 = r2_score(y_train, random_forest.predict(X_train))
forest_predictions = random_forest.predict(X_test)
forest_test_r2 = r2_score(y_test, forest_predictions)
forest_mae = mean_absolute_error(y_test, forest_predictions)
forest_rmse = root_mean_squared_error(y_test, forest_predictions)

page_width = 420 / 25.4
page_height = 594 / 25.4
fig = plt.figure(figsize=(page_width, page_height), facecolor="white")

margin = 0.028
full_w = 1 - 2 * margin
horizontal_gap = 0.016
vertical_gap = 0.006

fig.text(
    0.5,
    0.970,
    "IMDb-Bewertungen verstehen und vorhersagen",
    ha="center",
    va="center",
    fontproperties=FONTS["bold"],
    fontsize=36,
    color=NAVY,
)
fig.text(
    0.5,
    0.946,
    "Mustafa Takirtas  ·  KI-Programmierung  ·  OTH Regensburg",
    ha="center",
    va="center",
    fontproperties=FONTS["regular"],
    fontsize=18,
    color=NAVY,
)
fig.text(
    0.5,
    0.925,
    "Datensatz: IMDb 5000 Movie Dataset (Kaggle) [1]  ·  Zielvariable: imdb_score  ·  Aufgabe: Regression",
    ha="center",
    va="center",
    fontproperties=FONTS["bold"],
    fontsize=18,
    color=ORANGE,
)
fig.lines.extend(
    [
        plt.Line2D(
            [margin, 0.065],
            [0.925, 0.925],
            transform=fig.transFigure,
            color=ORANGE,
            lw=1.5,
        ),
        plt.Line2D(
            [0.935, 1 - margin],
            [0.925, 0.925],
            transform=fig.transFigure,
            color=ORANGE,
            lw=1.5,
        ),
    ]
)

row1_y = 0.830
row1_h = 0.080
col_w = (full_w - horizontal_gap) / 2

info = add_card(fig, [margin, row1_y, col_w, row1_h])
heading(info, "1", "Grundinformationen", 20.5)
info_text = (
    f"•  {rows} Filme, {columns} Spalten, {removed_duplicates} Duplikate entfernt\n"
    f"•  {numeric_count} numerisch, {text_count} textuell (Metadaten, Reichweite)\n"
    f"•  imdb_score: {de(stats['min'], 1)}-{de(stats['max'], 1)} · MW {de(stats['mean'])} · Median {de(stats['50%'])} · 0 NaN\n"
    "•  Verteilung leicht linksschief (MW < Median)"
)
body(info, info_text, 0.055, 0.69, 18.0, 1.16)

question = add_card(fig, [margin + col_w + horizontal_gap, row1_y, col_w, row1_h])
heading(question, "2", "Forschungsfrage", 20.5)
question_text = (
    "•  Welche Merkmale hängen mit dem IMDb-Score\n"
    "   zusammen, und wie gut lässt er sich vorhersagen?\n"
    "•  Ansatz: EDA → Korrelation → Split → Modellvergleich"
)
body(question, question_text, 0.055, 0.69, 18.0, 1.18)

eda_h = 0.158
eda_y = row1_y - vertical_gap - eda_h
eda = add_card(fig, [margin, eda_y, full_w, eda_h])
heading(eda, "3", "Explorative Datenanalyse", 20.5, x=0.025)
eda_text = (
    "•  85,5 % der Scores liegen zwischen 5 und 8.\n"
    "•  Die Verteilung ist linksschief (Schiefe = -0,74).\n"
    f"•  Fehlende Werte: gross {de(missing_percent['gross'], 1)} %, "
    f"budget {de(missing_percent['budget'], 1)} %.\n"
    "•  imdb_score enthält keine fehlenden Werte."
)
body(eda, eda_text, 0.030, 0.70, 18.0, 1.18)

hist_ax = fig.add_axes(
    [margin + full_w * 0.46, eda_y + 0.030, full_w * 0.50, eda_h - 0.050]
)
sns.histplot(
    df[TARGET],
    kde=True,
    bins=32,
    ax=hist_ax,
    color="#87B4D4",
    edgecolor=NAVY,
    linewidth=0.5,
    alpha=0.82,
)
if hist_ax.lines:
    hist_ax.lines[0].set_color(BLUE)
    hist_ax.lines[0].set_linewidth(2.0)
hist_ax.set_title(
    "Verteilung der IMDb-Bewertungen",
    fontproperties=FONTS["bold"],
    fontsize=16,
    color=NAVY,
    pad=6,
)
hist_ax.set_xlabel("IMDb-Score", fontproperties=FONTS["regular"], fontsize=13.5)
hist_ax.set_ylabel("Anzahl Filme", fontproperties=FONTS["regular"], fontsize=13.5)
style_axis(hist_ax)
eda.text(
    0.70,
    0.028,
    "Abb. 1 - Verteilung der IMDb-Bewertungen",
    ha="center",
    va="bottom",
    fontproperties=FONTS["italic"],
    fontsize=13.5,
    color=NAVY,
)

corr_h = 0.195
corr_extra_bottom = 0.025
corr_card_h = corr_h + corr_extra_bottom
corr_card_y = eda_y - vertical_gap - corr_card_h
corr_y = corr_card_y + corr_extra_bottom

corr_card = add_card(fig, [margin, corr_card_y, full_w, corr_card_h])

bar_ax = fig.add_axes(
    [margin + full_w * 0.145, corr_card_y + 0.047, full_w * 0.40, corr_card_h - 0.077]
)
bar_positions = np.arange(len(display_correlations))
bar_values = display_correlations.to_numpy()
bar_labels = [correlation_labels.get(name, name) for name in display_correlations.index]
bar_colors = [ORANGE if value < 0 else BLUE for value in bar_values]
bar_ax.barh(bar_positions, bar_values, color=bar_colors, alpha=0.88)
bar_ax.axvline(0, color=GRID, linewidth=1.2)
bar_ax.set_yticks(bar_positions, labels=bar_labels)
bar_ax.set_xlim(-0.26, 0.46)
bar_ax.set_xlabel("Pearson-Korrelation mit imdb_score", fontsize=14)
bar_ax.set_title(
    "Stärkste lineare Zusammenhänge",
    fontproperties=FONTS["bold"],
    fontsize=16,
    color=NAVY,
    pad=7,
)
for position, value in zip(bar_positions, bar_values):
    positive = value >= 0
    bar_ax.text(
        value + 0.012,
        position,
        de(value),
        ha="left",
        va="center",
        fontproperties=FONTS["bold"],
        fontsize=13,
        color=TEXT if positive else "white",
    )
style_axis(bar_ax)
bar_ax.tick_params(axis="x", labelsize=13.5)
bar_ax.tick_params(axis="y", labelsize=13.5, length=0)
corr_card.text(
    0.275,
    0.012,
    "Abb. 2 - Pearson-Korrelationen der numerischen Merkmale",
    ha="center",
    va="bottom",
    fontproperties=FONTS["italic"],
    fontsize=13.5,
    color=NAVY,
)

corr_text_ax = fig.add_axes(
    [margin + full_w * 0.61, corr_y + 0.012, full_w * 0.37, corr_h - 0.024]
)
corr_text_ax.axis("off")
heading(corr_text_ax, "4", "Korrelationsanalyse", 20.5, x=0.02)
corr_text = (
    "•  Stärkste lineare Zusammenhänge (Pearson):\n"
    f"   - Nutzerstimmen ({de(target_corr['num_voted_users'])})\n"
    f"   - Kritikerrezensionen ({de(target_corr['num_critic_for_reviews'])})\n"
    f"   - Nutzerrezensionen ({de(target_corr['num_user_for_reviews'])})\n\n"
    f"•  budget: kein relevanter linearer\n"
    f"   Zusammenhang ({de(target_corr['budget'])}).\n\n"
    f"•  title_year: leicht negativ ({de(target_corr['title_year'])});\n"
    "   spätere Jahre gehen mit etwas niedrigeren\n"
    "   Scores einher. Keine Kausalaussage."
)
body(corr_text_ax, corr_text, 0.04, 0.72, 18.0, 1.12)

model_h = 0.172
model_y = corr_card_y - vertical_gap - model_h
model_card = add_card(fig, [margin, model_y, full_w, model_h])
heading(model_card, "5", "Modell & Bewertung", 20.5, x=0.025)
model_text = (
    f"•  Vorverarbeitung: Duplikate entfernt; vollständige\n"
    f"   numerische Fälle → {X.shape[0]} Filme, {X.shape[1]} Merkmale\n"
    "•  StandardScaler für Linear; RF benötigt keine Skalierung\n"
    f"•  Split 60/20/20: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]} (seed=42)\n"
    f"•  Validierung R²: Linear {de(linear_val_r2)} · RF {de(forest_val_r2)}\n"
    "•  Final: Random Forest (beste Validierung)\n"
    "   n_estimators=100 · max_depth=None\n"
    "   min_samples_split=2 · random_state=42\n"
    f"•  Test: R² {de(forest_test_r2)} · MAE {de(forest_mae)} · RMSE {de(forest_rmse)} Scorepunkte\n"
    f"•  Train R² {de(forest_train_r2)} → Überanpassung; Val {de(forest_val_r2)} ≈ Test {de(forest_test_r2)}"
)
body(model_card, model_text, 0.030, 0.68, 18.0, 0.98)

scatter_ax = fig.add_axes(
    [margin + full_w * 0.56, model_y + 0.030, full_w * 0.40, model_h - 0.050]
)
scatter_ax.scatter(
    y_test, forest_predictions, s=18, alpha=0.33, color="#4A91C6", edgecolors="none"
)
min_value = min(float(y_test.min()), float(forest_predictions.min()))
max_value = max(float(y_test.max()), float(forest_predictions.max()))
scatter_ax.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    color="red",
    linewidth=1.7,
)
scatter_ax.set_title(
    "Vorhergesagt vs. tatsächlich (Random Forest)",
    fontproperties=FONTS["bold"],
    fontsize=16,
    color=NAVY,
    pad=5,
)
scatter_ax.set_xlabel(
    "Tatsächlicher IMDb-Score", fontproperties=FONTS["regular"], fontsize=13.5
)
scatter_ax.set_ylabel(
    "Vorhergesagter IMDb-Score", fontproperties=FONTS["regular"], fontsize=13.5
)
style_axis(scatter_ax)
model_card.text(
    0.73,
    0.020,
    "Abb. 3 - Vorhergesagt vs. Tatsächlich",
    ha="center",
    va="bottom",
    fontproperties=FONTS["italic"],
    fontsize=13.5,
    color=NAVY,
)

bottom_h = 0.095
bottom_y = model_y - vertical_gap - bottom_h
bottom_col_w = (full_w - horizontal_gap) / 2

learning = add_card(fig, [margin, bottom_y, bottom_col_w, bottom_h])
heading(learning, "6", "Fehlschläge, Lernmomente & Ausblick", 17.8)
learning_text = (
    "•  Nur numerische Merkmale; vollständige\n"
    "   Fallanalyse verliert 24,6 % der Filme.\n"
    "•  num_voted_users und gross entstehen erst\n"
    "   nach Veröffentlichung: keine Pre-Release-Prognose.\n"
    "•  Nächste Schritte: kategoriale Merkmale,\n"
    "   Imputation und Kreuzvalidierung."
)
body(learning, learning_text, 0.050, 0.74, 18.0, 1.00)

conclusion = add_card(
    fig, [margin + bottom_col_w + horizontal_gap, bottom_y, bottom_col_w, bottom_h]
)
heading(conclusion, "7", "Fazit", 20.5)
conclusion_text = (
    "•  Nutzerstimmen: stärkster linearer Zusammenhang.\n"
    "•  Random Forest besser als lineare Basis;\n"
    "   Test-R² = 0,51.\n"
    "•  Post-Release-Merkmale: keine Vorabprognose."
)
body(conclusion, conclusion_text, 0.055, 0.74, 18.0, 1.03)

sources_h = 0.143
sources_y = bottom_y - vertical_gap - sources_h
sources = add_card(fig, [margin, sources_y, full_w, sources_h])
heading(sources, "8", "Quellen (MLA, inkl. KI-Nutzung)", 18.5, y=0.89)
source_font = 18.0
source_spacing = 0.95 * TEXT_LINE_SPACING_SCALE
sources.text(
    0.025,
    0.63,
    "[1] IMDB 5000 Movie\n"
    "Dataset. Kaggle.\n"
    "kaggle.com/datasets/\n"
    "carolzhangdc/imdb-5000-\n"
    "movie-dataset. Zugriff:\n"
    "23. Juni 2026.",
    fontproperties=FONTS["regular"],
    fontsize=source_font,
    color=NAVY,
    va="top",
    linespacing=source_spacing,
)
sources.text(
    0.270,
    0.63,
    "[2] Anthropic. Claude\n"
    "[KI-Sprachmodell].\n"
    "claude.ai. Unterstützung\n"
    "bei Programmierung und\n"
    "Konzepterklärung.",
    fontproperties=FONTS["regular"],
    fontsize=source_font,
    color=NAVY,
    va="top",
    linespacing=source_spacing,
)
sources.text(
    0.515,
    0.63,
    "[3] Pedregosa, F., et al.\n"
    "Scikit-learn: Machine\n"
    "Learning in Python. JMLR,\n"
    "Bd. 12, 2011,\n"
    "S. 2825-30.",
    fontproperties=FONTS["regular"],
    fontsize=source_font,
    color=NAVY,
    va="top",
    linespacing=source_spacing,
)
sources.text(
    0.760,
    0.63,
    "[4] Waskom, M. seaborn:\n"
    "statistical data\n"
    "visualization. JOSS,\n"
    "Bd. 6, Nr. 60, 2021,\n"
    "S. 3021.",
    fontproperties=FONTS["regular"],
    fontsize=source_font,
    color=NAVY,
    va="top",
    linespacing=source_spacing,
)

if sources_y < 0:
    raise ValueError(
        "The vertical layout extends beyond the page. Reduce section heights or spacing."
    )

pdf_path = out_dir / "imdb_rating_prediction_poster.pdf"
fig.savefig(pdf_path, dpi=300, facecolor="white")
plt.close(fig)

print(pdf_path)
