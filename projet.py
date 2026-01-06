import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, title, filename, labels=("0", "1")):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)  # pas de couleur fixée
    plt.title(title)
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)

    # afficher les valeurs dans les cases
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_top_importances(importances_series, topk, title, filename):
    top = importances_series.sort_values(ascending=False).head(topk).sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(top.index, top.values)
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()







PATH = "C://Users//eyaja//Desktop//projet hayet//AMYLOSE3 (1).xls"  
df = pd.read_excel(PATH)
print("Shape :", df.shape)
print("Colonnes :", list(df.columns))

df["ETAT AUX DN"] = df["ETAT AUX DN"].astype(str).str.strip().str.upper()
end_stage_labels = {"IRCT", "HD"}
df["progression"] = df["ETAT AUX DN"].apply(lambda x: 1 if x in end_stage_labels else 0)

print("\nDistribution cible (progression):")
print(df["progression"].value_counts(dropna=False))



cols_to_drop = [
    "ETAT AUX DN",
    "DATE DERNIERES NOUVELLES",
    "DELAI SUIVI",
    "ANNEE DN",
    "progression"
]

cols_to_drop = [c for c in cols_to_drop if c in df.columns]

X = df.drop(columns=cols_to_drop)
y = df["progression"].astype(int)

print("\nX shape:", X.shape, "  y shape:", y.shape)



num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print("\nNb variables numériques:", len(num_cols))
print("Nb variables catégorielles:", len(cat_cols))
print("Exemples cat:", cat_cols[:10])


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 4) Modèle 1 : Régression Logistique (baseline clinique)
# =========================
from sklearn.linear_model import LogisticRegression

clf_lr = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=2000))
])

clf_lr.fit(X_train, y_train)

proba_lr = clf_lr.predict_proba(X_test)[:, 1]
pred_lr  = (proba_lr >= 0.5).astype(int)

plot_confusion_matrix(
    y_test, pred_lr,
    title="Matrice de confusion — Logistic Regression (seuil=0.50)",
    filename="CM_LogReg.png"
)


auc_lr = roc_auc_score(y_test, proba_lr)
print("\n===== Logistic Regression =====")
print("AUC =", round(auc_lr, 3))
print(classification_report(y_test, pred_lr))
print("Confusion matrix:\n", confusion_matrix(y_test, pred_lr))


# =========================
# 5) Modèle 2 : Random Forest (non-linéaire, robuste)
# =========================
from sklearn.ensemble import RandomForestClassifier

clf_rf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced"  
    ))
])

clf_rf.fit(X_train, y_train)

proba_rf = clf_rf.predict_proba(X_test)[:, 1]
pred_rf  = (proba_rf >= 0.5).astype(int)

plot_confusion_matrix(
    y_test, pred_rf,
    title="Matrice de confusion — Random Forest (seuil=0.50)",
    filename="CM_RF_0p50.png"
)

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def eval_threshold(y_true, proba, t):
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    f1 = f1_score(y_true, pred)
    youden = sens + spec - 1
    return {
        "threshold": t,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "youdenJ": youden,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn
    }

# 1) Balayer une grille de seuils
thresholds = np.linspace(0.05, 0.95, 181)
rows = [eval_threshold(y_test, proba_rf, t) for t in thresholds]
df_thr = pd.DataFrame(rows)

# 2) Seuils "optimaux" selon différents objectifs
best_youden = df_thr.loc[df_thr["youdenJ"].idxmax()]
best_f1 = df_thr.loc[df_thr["f1"].idxmax()]

# 3) Optimiser sensibilité avec contrainte sur la spécificité
# (tu peux changer 0.70 en 0.75 si tu veux être plus strict)
MIN_SPEC = 0.70
df_valid = df_thr[df_thr["specificity"] >= MIN_SPEC]
best_sens_constrained = df_valid.loc[df_valid["sensitivity"].idxmax()] if len(df_valid) > 0 else None

# 4) Baseline seuil=0.5 pour comparaison
baseline_05 = eval_threshold(y_test, proba_rf, 0.5)

print("\n=== Random Forest : optimisation du seuil (jeu de test) ===")
print("\n--- Baseline seuil=0.50 ---")
print(pd.Series(baseline_05).round(3))

print("\n--- Meilleur compromis (Youden J) ---")
print(pd.Series(best_youden).round(3))

print("\n--- Meilleur équilibre (F1) ---")
print(pd.Series(best_f1).round(3))

print(f"\n--- Sensibilité max avec spécificité >= {MIN_SPEC:.2f} ---")
if best_sens_constrained is not None:
    print(pd.Series(best_sens_constrained).round(3))
else:
    print("Aucun seuil ne respecte la contrainte de spécificité.")

# 5) Afficher les matrices de confusion pour les seuils choisis
def print_cm(y_true, proba, t, title):
    pred = (proba >= t).astype(int)
    cm = confusion_matrix(y_true, pred)
    print(f"\nMatrice de confusion - {title} (seuil={t:.2f})")
    print(cm)

print_cm(y_test, proba_rf, 0.5, "Baseline")
print_cm(y_test, proba_rf, float(best_youden["threshold"]), "Youden J")
print_cm(y_test, proba_rf, float(best_f1["threshold"]), "F1")

if best_sens_constrained is not None:
    print_cm(y_test, proba_rf, float(best_sens_constrained["threshold"]), f"Sens max (Spec>={MIN_SPEC:.2f})")


auc_rf = roc_auc_score(y_test, proba_rf)
print("\n===== Random Forest =====")
print("AUC =", round(auc_rf, 3))
print(classification_report(y_test, pred_rf))
print("Confusion matrix:\n", confusion_matrix(y_test, pred_rf))

t_youden = float(best_youden["threshold"])
pred_rf_youden = (proba_rf >= t_youden).astype(int)
plot_confusion_matrix(
    y_test, pred_rf_youden,
    title=f"Matrice de confusion — RF (Youden J, seuil={t_youden:.2f})",
    filename="CM_RF_Youden.png"
)

t_f1 = float(best_f1["threshold"])
pred_rf_f1 = (proba_rf >= t_f1).astype(int)
plot_confusion_matrix(
    y_test, pred_rf_f1,
    title=f"Matrice de confusion — RF (F1, seuil={t_f1:.2f})",
    filename="CM_RF_F1.png"
)

if best_sens_constrained is not None:
    t_c = float(best_sens_constrained["threshold"])
    pred_rf_c = (proba_rf >= t_c).astype(int)
    plot_confusion_matrix(
        y_test, pred_rf_c,
        title=f"Matrice de confusion — RF (Sens max, Spec>={MIN_SPEC:.2f}, seuil={t_c:.2f})",
        filename="CM_RF_SensMax_SpecConstraint.png"
    )



# =========================
# 8) Calibration des probabilités (Random Forest)
# =========================

# Calibration par Platt scaling (sigmoid)
clf_rf_cal = CalibratedClassifierCV(
    estimator=clf_rf,
    method="sigmoid",
    cv=5
)

clf_rf_cal.fit(X_train, y_train)

# Probabilités calibrées
proba_rf_cal = clf_rf_cal.predict_proba(X_test)[:, 1]

# AUC avant / après calibration
auc_uncal = roc_auc_score(y_test, proba_rf)
auc_cal = roc_auc_score(y_test, proba_rf_cal)

print("\n=== Calibration Random Forest ===")
print(f"AUC avant calibration : {auc_uncal:.3f}")
print(f"AUC après calibration : {auc_cal:.3f}")

# Courbes de calibration
frac_pos_uncal, mean_pred_uncal = calibration_curve(
    y_test, proba_rf, n_bins=10, strategy="uniform"
)

frac_pos_cal, mean_pred_cal = calibration_curve(
    y_test, proba_rf_cal, n_bins=10, strategy="uniform"
)

plt.figure(figsize=(6,6))
plt.plot(mean_pred_uncal, frac_pos_uncal, "o-", label="RF non calibré")
plt.plot(mean_pred_cal, frac_pos_cal, "o-", label="RF calibré")
plt.plot([0,1], [0,1], "--", color="gray")
plt.xlabel("Probabilité prédite")
plt.ylabel("Proportion observée d’IRCT/HD")
plt.title("Courbe de calibration – Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig("Calibration_RF.png", dpi=200)
plt.close()

print("Figure de calibration enregistrée : Calibration_RF.png")

# =========================
# 5bis) Modèle 3 : Gradient Boosting
# =========================
from sklearn.ensemble import GradientBoostingClassifier

clf_gb = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", GradientBoostingClassifier(
        random_state=42
    ))
])

clf_gb.fit(X_train, y_train)

proba_gb = clf_gb.predict_proba(X_test)[:, 1]
pred_gb  = (proba_gb >= 0.5).astype(int)

plot_confusion_matrix(
    y_test, pred_gb,
    title="Matrice de confusion — Gradient Boosting (seuil=0.50)",
    filename="CM_GradBoost.png"
)


auc_gb = roc_auc_score(y_test, proba_gb)
print("\n===== Gradient Boosting =====")
print("AUC =", round(auc_gb, 3))
print(classification_report(y_test, pred_gb))
print("Confusion matrix:\n", confusion_matrix(y_test, pred_gb))


# =========================
# Calcul des courbes ROC
# =========================
from sklearn.metrics import roc_curve

fpr_lr, tpr_lr, _ = roc_curve(y_test, proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)
fpr_gb, tpr_gb, _ = roc_curve(y_test, proba_gb)

# 6) Courbe ROC (LR, RF, GB) — sauvegarde
# =========================
import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr_lr, tpr_lr, label=f"LogReg AUC={auc_lr:.2f}")
plt.plot(fpr_rf, tpr_rf, label=f"RF AUC={auc_rf:.2f}")
plt.plot(fpr_gb, tpr_gb, label=f"GB AUC={auc_gb:.2f}")
plt.plot([0,1], [0,1], "--")
plt.xlabel("1 - Spécificité (FPR)")
plt.ylabel("Sensibilité (TPR)")
plt.legend()
plt.title("ROC - Amylose rénale (progression IRCT/HD)")
plt.tight_layout()
plt.savefig("ROC_amylose.png", dpi=200)
plt.close()

print("Figure ROC enregistrée : ROC_amylose.png")


# =========================
# 7) Importance des variables (Permutation Importance - fonctionne même avec OneHot)
# =========================
from sklearn.inspection import permutation_importance

# On calcule l'importance sur le modèle RF
result = permutation_importance(clf_rf, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc")

importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)

print("\nTop 15 variables importantes (Permutation Importance, RF):")
print(importances.head(15))

plot_top_importances(
    importances,
    topk=15,
    title="Top 15 variables importantes — RF (Permutation importance, AUC)",
    filename="Top15_Variables_Importantes_RF.png"
)


results = pd.DataFrame({
    "Modele": ["LogReg", "RandomForest", "GradientBoosting"],
    "AUC": [auc_lr, auc_rf, auc_gb]
}).sort_values("AUC", ascending=False)

print("\n=== Comparaison des AUC ===")
print(results)


def cross_validate_model(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    aucs, sens, specs, f1s = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_te, proba))
        sens.append(recall_score(y_te, pred))                 # sensibilité
        specs.append(recall_score(y_te, pred, pos_label=0))  # spécificité
        f1s.append(f1_score(y_te, pred))

    return {
        "AUC_mean": np.mean(aucs),
        "AUC_std": np.std(aucs),
        "Sens_mean": np.mean(sens),
        "Sens_std": np.std(sens),
        "Spec_mean": np.mean(specs),
        "Spec_std": np.std(specs),
        "F1_mean": np.mean(f1s),
        "F1_std": np.std(f1s)
    }
cv_lr = cross_validate_model(clf_lr, X, y)
cv_rf = cross_validate_model(clf_rf, X, y)
cv_gb = cross_validate_model(clf_gb, X, y)

cv_results = pd.DataFrame.from_dict({
    "Logistic Regression": cv_lr,
    "Random Forest": cv_rf,
    "Gradient Boosting": cv_gb
}, orient="index")

print("\n=== Validation croisée (5-fold) ===")
print(cv_results.round(3))

# =========================
# 9) Stratification du risque clinique
# =========================

# Création d’un DataFrame patient-test
df_risk = X_test.copy()
df_risk["proba_IRCT_HD"] = proba_rf_cal
df_risk["event_IRCT_HD"] = y_test.values

# Catégorisation du risque
def risk_group(p):
    if p < 0.30:
        return "Faible"
    elif p < 0.60:
        return "Intermédiaire"
    else:
        return "Élevé"

df_risk["groupe_risque"] = df_risk["proba_IRCT_HD"].apply(risk_group)

# Distribution des groupes
print("\n=== Répartition des groupes de risque ===")
print(df_risk["groupe_risque"].value_counts())

# Taux réel de progression par groupe
risk_summary = df_risk.groupby("groupe_risque").agg(
    N=("event_IRCT_HD", "count"),
    Taux_progression=("event_IRCT_HD", "mean")
)

print("\n=== Risque observé par groupe ===")
print((risk_summary * [1, 100]).round(1))

# --- S'assurer de l'ordre clinique des groupes ---
order = ["Faible", "Intermédiaire", "Élevé"]

# Re-index pour garder l'ordre
risk_summary_ordered = risk_summary.reindex(order)

# 1) Barplot : taux observé (%) de progression par groupe
plt.figure(figsize=(6,4))
plt.bar(risk_summary_ordered.index, 100*risk_summary_ordered["Taux_progression"])
plt.ylabel("Taux observé de progression IRCT/HD (%)")
plt.xlabel("Groupe de risque")
plt.title("Progression observée selon le groupe de risque")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("Risque_observe_par_groupe.png", dpi=200)
plt.close()
print("Figure enregistrée : Risque_observe_par_groupe.png")

# 2) Barplot : effectifs par groupe
counts = df_risk["groupe_risque"].value_counts().reindex(order)

plt.figure(figsize=(6,4))
plt.bar(counts.index, counts.values)
plt.ylabel("Nombre de patients (N)")
plt.xlabel("Groupe de risque")
plt.title("Répartition des patients selon le groupe de risque")
plt.tight_layout()
plt.savefig("Effectifs_par_groupe.png", dpi=200)
plt.close()
print("Figure enregistrée : Effectifs_par_groupe.png")

# 3) Histogrammes : distribution des probabilités (calibrées) selon événement
plt.figure(figsize=(7,4))

proba0 = df_risk.loc[df_risk["event_IRCT_HD"] == 0, "proba_IRCT_HD"].values
proba1 = df_risk.loc[df_risk["event_IRCT_HD"] == 1, "proba_IRCT_HD"].values

plt.hist(proba0, bins=10, alpha=0.7, label="Pas de progression (0)")
plt.hist(proba1, bins=10, alpha=0.7, label="Progression IRCT/HD (1)")

plt.xlabel("Probabilité prédite (calibrée)")
plt.ylabel("Nombre de patients")
plt.title("Distribution des probabilités prédites selon l’événement")
plt.legend()
plt.tight_layout()
plt.savefig("Distribution_probabilites_par_evenement.png", dpi=200)
plt.close()
print("Figure enregistrée : Distribution_probabilites_par_evenement.png")

# 4) Boxplot : probabilités par groupe de risque (très utile en rapport)
plt.figure(figsize=(7,4))

data_box = [df_risk.loc[df_risk["groupe_risque"] == g, "proba_IRCT_HD"].values for g in order]
plt.boxplot(data_box, labels=order, showmeans=True)
plt.ylabel("Probabilité prédite (calibrée)")
plt.xlabel("Groupe de risque")
plt.title("Probabilités prédites selon le groupe de risque")
plt.tight_layout()
plt.savefig("Boxplot_probabilites_par_groupe.png", dpi=200)
plt.close()
print("Figure enregistrée : Boxplot_probabilites_par_groupe.png")



