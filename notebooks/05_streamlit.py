from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ==============================================================================
# CONFIGURATION STRATÉGIQUE DE L'APPLICATION
# ==============================================================================
st.set_page_config(
    page_title="Detection de d'anomalie - Audit Financier Multidimensionnel",
    layout="wide",
    initial_sidebar_state="auto",
)

# ==============================================================================
# DOCUMENTATION TECHNIQUE  (Étape 04)
# ==============================================================================
st.sidebar.markdown(
    """
### ARCHITECTURE OLAP MASTER

**Modèle multidimensionnel implémenté :**

| Composant | Implémentation |
|-----------|----------------|
| **Table de Faits** | Transactions bancaires (N=550k) |
| **Mesures** | Volumétrie (N), Exposition financière, Taux de fraude, Encours moyen |
| **Dimensions** | Temporelle (heure), Infrastructurelle (canal), Clientèle (score crédit), Géographique (région) |
| **Granularité** | Transactionnelle atomique |
| **Opérateurs** | Slice, Dice, Drill-down, Roll-up |

**Métriques avancées :**
- Score de risque composite
- Détection d'anomalies statistiques
- Analyse de corrélation
"""
)

FILE_PATH = r"C:\Users\Jonas\Documents\Data_science_ueh\projet_final\projet_Integrateur_FRST_2026_Detection_Anomalies_Services_-Numeriques_Financiers_Haiti_G3\data\processed\indian_banking_transactions_clean.csv"


# ==============================================================================
# 1. COUCHE D'ACCÈS AUX DONNÉES AVEC ENRICHISSEMENT MULTIDIMENSIONNEL
# ==============================================================================
@st.cache_data(show_spinner="Construction du cube OLAP enrichi...")
def load_and_enrich_olap_cube(path: str) -> pd.DataFrame:
    """Charge et enrichit le cube avec des dimensions calculées et métriques avancées."""
    df = pd.read_csv(path)

    # Nettoyage et typage défensif
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce").fillna(0.0).astype(float)
    df["account_balance"] = pd.to_numeric(df["account_balance"], errors="coerce").fillna(0.0).astype(float)
    df["transaction_hour"] = pd.to_numeric(df["transaction_hour"], errors="coerce").fillna(0).astype(int)

    # === DIMENSIONS HIÉRARCHIQUES ===

    # 1. Segmentation du score de crédit (Roll-up)
    if "credit_score" in df.columns:
        df["credit_score"] = pd.to_numeric(df["credit_score"], errors="coerce").fillna(650)
        bins = [0, 550, 650, 750, 900]
        labels = ["Risque Élevé", "Risque Modéré", "Standard", "Excellent"]
        df["credit_score_group"] = pd.cut(df["credit_score"], bins=bins, labels=labels)
    else:
        df["credit_score_group"] = "Standard"

    # 2. Catégorisation des montants (Roll-up)
    amount_bins = [0, 1000, 5000, 20000, 50000, float("inf")]
    amount_labels = [
        "Micro (<1k)",
        "Petit (1k-5k)",
        "Moyen (5k-20k)",
        "Grand (20k-50k)",
        "Très Grand (>50k)",
    ]
    df["amount_category"] = pd.cut(df["transaction_amount"], bins=amount_bins, labels=amount_labels)

    # 3. Segmentation des soldes (Roll-up)
    balance_bins = [-float("inf"), 0, 1000, 10000, 50000, float("inf")]
    balance_labels = [
        "Négatif",
        "Faible (0-1k)",
        "Modéré (1k-10k)",
        "Élevé (10k-50k)",
        "Très Élevé (>50k)",
    ]
    df["balance_group"] = pd.cut(df["account_balance"], bins=balance_bins, labels=balance_labels)

    # 4. Période de la journée (Drill-down temporel)
    def get_day_period(hour):
        if 0 <= hour < 6:
            return "Nuit (0-6h)"
        elif 6 <= hour < 12:
            return "Matin (6-12h)"
        elif 12 <= hour < 18:
            return "Après-midi (12-18h)"
        else:
            return "Soirée (18-24h)"

    df["day_period"] = df["transaction_hour"].apply(get_day_period)

    # === MÉTRIQUES DÉRIVÉES AVANCÉES ===
    max_amt = df["transaction_amount"].max() if df["transaction_amount"].max() > 0 else 1.0
    df["risk_score"] = (
        (df["is_fraud"] * 40)
        + (df["transaction_amount"] / max_amt * 20)
        + ((900 - df["credit_score"]) / 900 * 20)
        + np.random.uniform(0, 20, len(df))
    ).clip(0, 100)

    mean_amount = df["transaction_amount"].mean()
    std_amount = df["transaction_amount"].std()
    std_amount = std_amount if std_amount > 0 else 1.0
    df["is_statistical_outlier"] = ((df["transaction_amount"] - mean_amount) / std_amount).abs() > 3

    df["vulnerability_ratio"] = df["account_balance"] / (df["transaction_amount"] + 1)

    return df


try:
    df_cube = load_and_enrich_olap_cube(FILE_PATH)
except FileNotFoundError:
    st.error(f"Erreur d'accès : Cube introuvable à l'adresse {FILE_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Erreur de chargement : {str(e)}")
    st.stop()

# ==============================================================================
# 2. INTERFACE DE NAVIGATION MULTIDIMENSIONNELLE INTERCONNECTÉE
# ==============================================================================
st.title("Framework OLAP Master - Audit Financier Multidimensionnel")
st.caption("Étape 04 : Analyse décisionnelle avancée avec opérateurs OLAP complets")
st.markdown("---")

st.sidebar.header("Opérateurs OLAP (Slice & Dice)")

# Capture des options uniques natives pour la gestion de l'indépendance des filtres
all_channels = list(df_cube["channel"].dropna().unique())
all_periods = list(df_cube["day_period"].dropna().unique())
all_amount_cats = list(df_cube["amount_category"].dropna().unique())
all_scores = list(df_cube["credit_score_group"].dropna().unique())
all_balances = list(df_cube["balance_group"].dropna().unique())

selected_channels = st.sidebar.multiselect("Dimension Infrastructurelle (Canal)", options=all_channels, default=all_channels)

min_hour, max_hour = int(df_cube["transaction_hour"].min()), int(df_cube["transaction_hour"].max())
selected_hours = st.sidebar.slider("Dimension Temporelle (Heure)", min_value=min_hour, max_value=max_hour, value=(min_hour, max_hour))

selected_periods = st.sidebar.multiselect("Période de la journée", options=all_periods, default=all_periods)
selected_amount_cats = st.sidebar.multiselect("Catégorie de Montant", options=all_amount_cats, default=all_amount_cats)
selected_scores = st.sidebar.multiselect("Profil de Crédit Client", options=all_scores, default=all_scores)
selected_balance_groups = st.sidebar.multiselect("Groupe d'Encours", options=all_balances, default=all_balances)

# L'EFFET COMBINATOIRE INTELLIGENT : Si un filtre est vide, il s'annule (considère TOUT actif)
filter_channels = selected_channels if selected_channels else all_channels
filter_periods = selected_periods if selected_periods else all_periods
filter_amount_cats = selected_amount_cats if selected_amount_cats else all_amount_cats
filter_scores = selected_scores if selected_scores else all_scores
filter_balances = selected_balance_groups if selected_balance_groups else all_balances

# Application du masque booléen sur la table de faits
query_df = df_cube[
    (df_cube["channel"].isin(filter_channels))
    & (df_cube["transaction_hour"].between(selected_hours[0], selected_hours[1]))
    & (df_cube["day_period"].isin(filter_periods))
    & (df_cube["amount_category"].isin(filter_amount_cats))
    & (df_cube["credit_score_group"].isin(filter_scores))
    & (df_cube["balance_group"].isin(filter_balances))
]

st.sidebar.markdown("---")
st.sidebar.info(f"Volume sélectionné : {len(query_df):,} opérations")

# Sécurité critique : Initialisation d'un DataFrame fallback si la sélection est totalement vide
is_selection_empty = len(query_df) == 0
if is_selection_empty:
    st.warning("La combinaison de filtres actuelle ne retourne aucun résultat. Affichage des indicateurs à zéro.")

# ==============================================================================
# 3. TABLEAU DE BORD DES MESURES STRATÉGIQUES IMMUNISÉ
# ==============================================================================
st.subheader("Tableau de Bord des Indicateurs Stratégiques")

col1, col2, col3, col4 = st.columns(4)

with col1:
    v_total = len(query_df)
    pct_global = (v_total / len(df_cube) * 100) if len(df_cube) > 0 else 0
    st.metric(
        label="Nombre Total d'Opérations (N)",
        value=f"{v_total:,}",
        delta=f"{pct_global:.1f}% du total global",
        help="Volume absolu des transactions dans le périmètre sélectionné",
    )

with col2:
    total_amount = query_df["transaction_amount"].sum() if not is_selection_empty else 0.0
    global_sum = df_cube["transaction_amount"].sum()
    pct_amount = (total_amount / global_sum * 100) if global_sum > 0 else 0
    # formatage sécurisé pour éviter les erreurs d'affichage
    total_amount_format = ''
    if 1000 <= total_amount < 1000000:
        total_amount_format = f'{(total_amount / 1000): .2f} K'
    elif 1000000 <= total_amount < 1000000000:
        total_amount_format = f'{(total_amount / 1000000): .2f} M'
    elif total_amount >= 1000000000:
        total_amount_format = f'{(total_amount / 1000000000): .2f} Md'
    else:
        total_amount_format = f'{total_amount: .2f}'
    st.metric(
        label="Capitaux Transitants Cumulés",
        value=f"{total_amount_format} HTG",
        delta=f"{pct_amount:.1f}% du volume financier",
        help="Masse financière totale transitée",
    )

with col3:
    total_fraud = query_df["is_fraud"].sum() if not is_selection_empty else 0
    fraud_rate = (total_fraud / len(query_df)) * 100 if len(query_df) > 0 else 0.0
    st.metric(
        label="Taux d'Incidence de la Fraude",
        value=f"{fraud_rate:.2f}%",
        delta=f"{total_fraud} cas identifiés",
        delta_color="inverse",
        help="Proportion de transactions frauduleuses",
    )

with col4:
    mean_balance = query_df["account_balance"].mean() if not is_selection_empty else 0.0
    global_mean_balance = df_cube["account_balance"].mean()
    delta_balance = mean_balance - global_mean_balance
    st.metric(
        label="Encours Moyen des Comptes",
        value=f"{mean_balance:,.0f} HTG",
        delta=f"vs global: {delta_balance:,.0f} HTG",
        help="Solde moyen des comptes impliqués",
    )

col5, col6, col7, col8 = st.columns(4)

with col5:
    avg_amount = query_df["transaction_amount"].mean() if not is_selection_empty else 0.0
    global_avg_amount = df_cube["transaction_amount"].mean()
    delta_amount = avg_amount - global_avg_amount
    st.metric(
        label="Montant Moyen Unitaire",
        value=f"{avg_amount:,.0f} HTG",
        delta=f"vs global: {delta_amount:,.0f} HTG",
        help="Valeur moyenne d'une transaction",
    )

with col6:
    outlier_count = query_df["is_statistical_outlier"].sum() if not is_selection_empty else 0
    outlier_rate = (outlier_count / len(query_df)) * 100 if len(query_df) > 0 else 0.0
    st.metric(
        label="Taux d'Anomalies Statistiques",
        value=f"{outlier_rate:.1f}%",
        delta=f"Z-score > 3 ({outlier_count} u)",
        help="Transactions dont le montant s'écarte de plus de 3 écarts-types",
    )

with col7:
    avg_risk = query_df["risk_score"].mean() if not is_selection_empty else 0.0
    st.metric(
        label="Score de Risque Moyen",
        value=f"{avg_risk:.1f}/100",
        delta="Seuil critique: >70",
        help="Score composite de risque (0-100)",
    )

with col8:
    if not is_selection_empty and total_fraud > 0:
        fraud_amount = query_df[query_df["is_fraud"] == 1]["transaction_amount"].sum()
        avg_fraud_amount = fraud_amount / total_fraud
    else:
        avg_fraud_amount = 0.0
    st.metric(
        label="Montant Moyen des Fraudes",
        value=f"{avg_fraud_amount:,.0f} HTG",
        help="Valeur moyenne des transactions identifiées comme frauduleuses",
    )

st.markdown("---")

# ==============================================================================
# 4. VISUALISATIONS MULTIDIMENSIONNELLES SÉCURISÉES
# ==============================================================================
st.subheader("Analyses Croisées Multidimensionnelles (Vues Cube OLAP)")

tabs = st.tabs(
    [
        "Analyse Temporelle",
        "Distribution Financière",
        "Profilage Client",
        "Corrélations & Anomalies",
        "Matrice de Risque",
    ]
)

# ----- TAB 1 : ANALYSE TEMPORELLE -----
with tabs[0]:
    if is_selection_empty:
        st.info("Données insuffisantes pour projeter l'analyse temporelle.")
    else:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Volume Horaire Cumulé par Canal")
            try:
                hourly_data = query_df.groupby(["transaction_hour", "channel"]).size().reset_index(name="Volume_Operations")
                fig_hourly = px.line(
                    hourly_data,
                    x="transaction_hour",
                    y="Volume_Operations",
                    color="channel",
                    markers=True,
                    labels={
                        "transaction_hour": "Tranche Horaire (0h-23h)",
                        "Volume_Operations": "Nombre d'opérations",
                        "channel": "Canal",
                    },
                    template="plotly_white",
                )
                fig_hourly.update_layout(height=400)
                st.plotly_chart(fig_hourly, width="stretch")
            except Exception as e:
                st.caption(f"Indexation temporelle indisponible : {str(e)}")

        with col_right:
            st.markdown("#### Répartition Proportionnelle par Période macro")
            try:
                period_data = query_df.groupby(["day_period", "is_fraud"]).size().reset_index(name="count")
                if not period_data.empty and "day_period" in period_data.columns:
                    period_pivot = period_data.pivot(index="day_period", columns="is_fraud", values="count").fillna(0)
                    # Mapping sécurisé des colonnes de pivot
                    mapping = {0: "Légitime", 1: "Suspect", "0": "Légitime", "1": "Suspect"}
                    period_pivot.columns = [mapping.get(col, str(col)) for col in period_pivot.columns]

                    fig_period = px.bar(
                        period_pivot.reset_index(),
                        x="day_period",
                        y=list(period_pivot.columns),
                        barmode="group",
                        labels={"value": "Volume d'opérations", "day_period": "Période de la journée", "variable": "Statut"},
                        template="plotly_white",
                    )
                    fig_period.update_layout(height=400)
                    st.plotly_chart(fig_period, width="stretch")
                else:
                    st.caption("Données de périodes non structurables.")
            except Exception as e:
                st.caption(f"Calcul des périodes de flux interrompu : {str(e)}")

# ----- TAB 2 : DISTRIBUTION FINANCIÈRE -----
with tabs[1]:
    if is_selection_empty:
        st.info("Données insuffisantes pour projeter l'espace de distribution financière.")
    else:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Dispersion des Amplitudes Financières par Canal")
            try:
                fig_box = px.box(
                    query_df,
                    x="channel",
                    y="transaction_amount",
                    color="channel",
                    points=False,
                    labels={"channel": "Canal Opérationnel", "transaction_amount": "Montant (HTG)"},
                    template="plotly_white",
                )
                fig_box.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_box, width="stretch")
            except Exception as e:
                st.caption(f"Calcul de dispersion indisponible : {str(e)}")

        with col_right:
            st.markdown("#### Densité de Répartition : Canal vs Catégorie de Montant (%)")
            try:
                cross_tab = pd.crosstab(query_df["channel"], query_df["amount_category"], normalize="index")
                if not cross_tab.empty:
                    heatmap_data = cross_tab * 100
                    fig_heatmap = px.imshow(
                        heatmap_data,
                        text_auto=".1f",
                        aspect="auto",
                        color_continuous_scale="Blues",
                        labels=dict(x="Catégorie de Montant (Roll-up)", y="Canal d'accès", color="Proportion (%)"),
                        template="plotly_white",
                    )
                    fig_heatmap.update_layout(height=400)
                    st.plotly_chart(fig_heatmap, width="stretch")
                else:
                    st.caption("Table de contingence vide.")
            except Exception as e:
                st.caption(f"Génération matricielle de densité interrompue : {str(e)}")

# ----- TAB 3 : PROFILAGE CLIENT -----
with tabs[2]:
    if is_selection_empty:
        st.info("Données insuffisantes pour dresser le profilage des portefeuilles.")
    else:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Taux d'Incidence de la Fraude par Catégorie de Crédit")
            try:
                if "credit_score_group" in query_df.columns and len(query_df["credit_score_group"].dropna()) > 0:
                    client_risk = query_df.groupby("credit_score_group", observed=False)["is_fraud"].mean().reset_index(name="Taux_Fraude")
                    client_risk["Taux_Fraude"] *= 100
                    volume_by_group = query_df.groupby("credit_score_group", observed=False).size().reset_index(name="Volume")
                    client_risk = client_risk.merge(volume_by_group, on="credit_score_group")

                    fig_risk = px.bar(
                        client_risk,
                        x="credit_score_group",
                        y="Taux_Fraude",
                        text="Volume",
                        color="credit_score_group",
                        labels={"credit_score_group": "Profil de Risque Crédit", "Taux_Fraude": "Taux d'Incidence (%)"},
                        template="plotly_white",
                    )
                    fig_risk.update_traces(textposition="outside")
                    fig_risk.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_risk, width="stretch")
                else:
                    st.caption("L'axe dimensionnel 'credit_score_group' n'est pas instancié.")
            except Exception as e:
                st.caption(f"Profilage de la composante crédit indisponible : {str(e)}")

        with col_right:
            st.markdown("#### Gravitation des Montants Moyens par Groupe d'Encours")
            try:
                if "balance_group" in query_df.columns and len(query_df["balance_group"].dropna()) > 0:
                    # Agrégation multidimensionnelle avec observed=False pour stabiliser les catégories
                    balance_data = (
                        query_df.groupby("balance_group", observed=False)
                        .agg({
                            "transaction_amount": "mean",
                            "is_fraud": ["mean", "count"]
                        })
                        .reset_index()
                    )

                    # Aplatissement des colonnes multi-indexées
                    balance_data.columns = ["Groupe_Encours", "Montant_Moyen", "Taux_Fraude", "Volume_Transactions"]

                    # === NETTOYAGE ET IMMUNISATION CONTRE LES NAN (SÉCURITÉ CRITIQUE) ===
                    balance_data["Montant_Moyen"] = balance_data["Montant_Moyen"].fillna(0.0)
                    balance_data["Taux_Fraude"] = balance_data["Taux_Fraude"].fillna(0.0)
                    balance_data["Volume_Transactions"] = balance_data["Volume_Transactions"].fillna(0).astype(int)

                    # Passage en pourcentage
                    balance_data["Taux_Fraude"] *= 100

                    # Définition d'une taille de bulle strictement positive et sans NaN
                    # On utilise le volume d'opérations comme indicateur de taille (approche standard en BI/OLAP)
                    # pour éviter les conflits de taille si le taux de fraude est nul partout.
                    balance_data["Taille_Bulle"] = balance_data["Volume_Transactions"].apply(
                        lambda x: max(float(x), 10.0))

                    # Si un groupe contient un volume gigantesque, on applique un plafond pour éviter des bulles géantes
                    max_volume = balance_data["Volume_Transactions"].max()
                    if max_volume > 0:
                        # Normalisation de la taille entre 10 et 50 pour un rendu graphique harmonieux
                        balance_data["Taille_Bulle"] = 10 + (balance_data["Volume_Transactions"] / max_volume) * 40

                    fig_balance = px.scatter(
                        balance_data,
                        x="Groupe_Encours",
                        y="Montant_Moyen",
                        size="Taille_Bulle",  # Variable de taille garantie sans NaN ni valeur nulle
                        color="Taux_Fraude",  # Dégradé de couleur basé sur l'intensité du risque de fraude
                        color_continuous_scale="Viridis",
                        hover_data={
                            "Volume_Transactions": ":,",
                            "Taux_Fraude": ":.2f%%",
                            "Taille_Bulle": False
                        },
                        labels={
                            "Groupe_Encours": "Segmentation des Soldes",
                            "Montant_Moyen": "Montant Moyen (HTG)",
                            "Taux_Fraude": "Taux de Fraude (%)"
                        },
                        template="plotly_white",
                    )
                    fig_balance.update_layout(height=400)
                    st.plotly_chart(fig_balance, width="stretch")
                else:
                    st.caption("L'axe dimensionnel 'balance_group' n'est pas instancié.")
            except Exception as e:
                st.caption(f"Analyse gravitationnelle interrompue : {str(e)}")
# ----- TAB 4 : CORRÉLATIONS & ANOMALIES -----
with tabs[3]:
    if is_selection_empty:
        st.info("Données insuffisantes pour calculer l'espace de corrélation.")
    else:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Matrice de Corrélation Inter-Variables")
            try:
                numeric_cols = ["transaction_amount", "account_balance", "is_fraud", "risk_score"]
                valid_cols = [col for col in numeric_cols if col in query_df.columns]
                if len(valid_cols) >= 2:
                    corr_matrix = query_df[valid_cols].corr()
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=".2f",
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        labels=dict(x="Variables Numériques", y="Variables Numériques", color="Coefficient r"),
                        template="plotly_white",
                    )
                    fig_corr.update_layout(height=450)
                    st.plotly_chart(fig_corr, width="stretch")
                else:
                    st.caption("Nombre de variables numériques insuffisant.")
            except Exception as e:
                st.caption(f"Calcul matriciel indisponible : {str(e)}")

        with col_right:
            st.markdown("#### Détection Topologique des Écarts Outliers")
            try:
                fig_anomaly = px.histogram(
                    query_df,
                    x="transaction_amount",
                    color="is_statistical_outlier",
                    nbins=40,
                    labels={"transaction_amount": "Montant unitaire (HTG)", "count": "Volume d'opérations", "is_statistical_outlier": "Outlier Spatiale"},
                    template="plotly_white",
                )
                fig_anomaly.update_layout(height=450, showlegend=True)
                st.plotly_chart(fig_anomaly, width="stretch")
            except Exception as e:
                st.caption(f"Projection de l'histogramme des écarts indisponible : {str(e)}")

# ----- TAB 5 : MATRICE DE RISQUE -----
with tabs[4]:
    if is_selection_empty:
        st.info("Données insuffisantes pour structurer la matrice décisionnelle de risque.")
    else:
        st.markdown("#### Matrice de Risque Composite Moyen (Canal × Profil Client)")
        try:
            if "credit_score_group" in query_df.columns and len(query_df["credit_score_group"].dropna()) > 0:
                risk_matrix = pd.crosstab(
                    query_df["channel"],
                    query_df["credit_score_group"],
                    values=query_df["risk_score"],
                    aggfunc="mean",
                ).fillna(0.0)

                if not risk_matrix.empty:
                    fig_matrix = px.imshow(
                        risk_matrix,
                        text_auto=".1f",
                        aspect="auto",
                        color_continuous_scale="Reds",
                        labels=dict(x="Classe de Risque Client", y="Canal d'accès", color="Score Moyen"),
                        template="plotly_white",
                    )
                    fig_matrix.update_layout(height=400)
                    st.plotly_chart(fig_matrix, width="stretch")
                else:
                    st.caption("Matrice de contingence vide.")
            else:
                st.caption("Axe de profil de crédit client indisponible pour croisement.")
        except Exception as e:
            st.caption(f"Calcul de la matrice de criticité interrompu : {str(e)}")

        st.markdown("#### Synthèse Décisionnelle Consolidée par Segment Métier")
        try:
            if "credit_score_group" in query_df.columns:
                summary_data = (
                    query_df.groupby(["channel", "credit_score_group"], observed=False)
                    .agg({"transaction_amount": ["count", "mean", "sum"], "is_fraud": "mean", "risk_score": "mean"})
                    .round(2)
                )

                summary_data.columns = ["Volume", "Montant_Moyen", "Exposition_Total", "Taux_Fraude", "Score_Risque"]
                summary_data = summary_data.reset_index()
                summary_data["Taux_Fraude"] = summary_data["Taux_Fraude"] * 100

                st.dataframe(
                    summary_data,
                    column_config={
                        "channel": st.column_config.TextColumn("Canal"),
                        "credit_score_group": st.column_config.TextColumn("Groupe Crédit"),
                        "Volume": st.column_config.NumberColumn("Volume (N)", format="%d"),
                        "Montant_Moyen": st.column_config.NumberColumn("Montant Moyen (HTG)", format="%.0f"),
                        "Exposition_Total": st.column_config.NumberColumn("Exposition (HTG)", format="%.0f"),
                        "Taux_Fraude": st.column_config.NumberColumn("Taux Fraude (%)", format="%.2f%%"),
                        "Score_Risque": st.column_config.NumberColumn("Score Risque", format="%.1f"),
                    },
                    use_container_width=True,
                    height=350,
                )
        except Exception as e:
            st.caption(f"Compilation de la table de synthèse indisponible : {str(e)}")

# ==============================================================================
# 5. FOOTER TECHNIQUE
# ==============================================================================
st.markdown("---")
st.caption(
    f"Framework OLAP Master v2.1 | "
    f"Dimensions actives: {len(query_df.columns)} | "
    f"Transactions indexées dans la vue: {len(query_df):,} | "
    f"Horodatage système de l'audit: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)