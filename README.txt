
================================================================================
PROJET DE DETECTION D'ANOMALIES DANS LES SERVICES FINANCIERS NUMERIQUES EN HAITI
================================================================================

IDENTIFICATION DU PROJET
========================

Titre           : Conception d'un système intelligent de détection d'anomalies 
                  dans les services financiers numériques en Haïti
Groupe          : G3
Membres         : Blemy JOSEPH, Jonas CLOCIN, Martin FRANÇOIS
Encadrant       : Evens TOUSSAINT
Date            : 26/06/2026
Domaine         : Secteur bancaire / Services financiers numériques


OBJECTIFS DU PROJET
===================

Ce projet vise à développer un système capable de :

    1. Détecter automatiquement les transactions suspectes
    2. Analyser les comportements habituels des utilisateurs
    3. Identifier les anomalies à risque
    4. Améliorer la sécurité des services financiers numériques en Haïti


STRUCTURE DU PROJET
===================

pwoje_deteksyon_anomali/
│
├── README.md                          # Description générale du projet
├── requirements.txt                   # Dépendances Python
├── .gitignore                         # Fichiers à ignorer
├── LICENSE                            # Licence MIT
│
├── data/
│   ├── raw/
│   │   └── indian_banking_transactions.csv    # Données brutes (550 000 lignes)
│   ├── processed/
│   │   ├── indian_banking_transactions_clean.csv   # Données nettoyées
│   │   ├── data_dictionary.csv                    # Dictionnaire des variables
│   │   ├── cleaning_rules.txt                     # Règles de nettoyage
│   │   └── cleaning_log.txt                       # Journal des opérations
│   └── README.md                         # Description des données
│
├── notebooks/
│   ├── 01_EDA.ipynb                     # Analyse exploratoire et préparation
│   ├── 02_isolation_forest.ipynb        # Modèle Isolation Forest
│   ├── 03_dbscan_acp.ipynb              # Modèle DBSCAN + ACP
│   └── 04_evaluation.ipynb              # Évaluation et comparaison
│
├── results/
│   ├── figures/
│   │   ├── distributions.png
│   │   ├── boxplots.png
│   │   ├── correlation_heatmap.png
│   │   └── topological_isolation.png
│   ├── metrics/
│   │   ├── descriptive_stats_numeric.csv
│   │   ├── correlation_matrix.csv
│   │   └── summary_report.txt
│   └── anomalies/
│       └── detected_anomalies.csv
│
│
├── configs/
│   └── config.yml                       # Paramètres des modèles
│
└── docs/
    ├── rapport_final.pdf                # Rapport 20 pages
    ├── presentation.pptx                # Présentation PowerPoint
    └── annexes/                         # Documents complémentaires


JEU DE DONNEES
==============

Source          : Indian Banking Transactions Dataset
Volume          : 550 000 transactions
Variables       : 20
Format          : CSV

Variables principales :
    - transaction_id      : Identifiant unique de la transaction
    - customer_id         : Identifiant unique du client
    - transaction_date    : Date de la transaction
    - transaction_time    : Heure de la transaction
    - transaction_amount  : Montant de la transaction
    - account_balance     : Solde après transaction
    - channel             : Canal de transaction (Mobile_App, Web, ATM, etc.)
    - is_fraud            : Indicateur de fraude (0 = Normal, 1 = Fraude)
    - transaction_hour    : Heure de la transaction (0-23)

    + 11 autres variables (voir data_dictionary.csv)


TRAITEMENTS EFFECTUES
=====================

1.  CHARGEMENT DES DONNEES
    - Fichier : indian_banking_transactions.csv
    - 550 000 lignes x 20 colonnes

2.  NETTOYAGE DES DONNEES
    - Aucun doublon détecté
    - Valeurs manquantes : loan_type (68.6%) remplacé par "Unknown"
    - Conversion des dates en datetime
    - Conservation des valeurs extrêmes (outliers)

3.  ANALYSE EXPLORATOIRE
    - Statistiques descriptives
    - Distributions (histogrammes, boxplots)
    - Corrélations (heatmap)
    - Détection préliminaire des anomalies
    - Analyse topologique (isolation des fraudes)
    - Analyse des risques par canal
    - Saisonnalité horaire

4.  RESULTATS CLES
    - Taux de fraude : 1% (5 500 transactions)
    - 35% des clients ont un prêt actif
    - Canal le plus utilisé : Mobile_App (40%)
    - Canal le plus risqué : API (15-20% de fraude)
    - Données synthétiques (activité constante sur 24h)


ALGORITHMES UTILISES
====================

1.  ISOLATION FOREST
    - Détection d'anomalies par isolation
    - Basé sur l'isolation spatiale des points aberrants
    - Adapté aux données déséquilibrées (1% de fraudes)

2.  DBSCAN
    - Clustering par densité
    - Identification des points aberrants (bruit)
    - Segmentation des comportements

3.  ACP (Analyse en Composantes Principales)
    - Réduction de dimensionnalité (7 → 2)
    - Visualisation des données


PRINCIPAUX RESULTATS DE L'ANALYSE
=================================

1.  VARIABILITE SIGNIFICATIVE
    - Montants : de 2.40 à 10 000 000
    - Présence de 7 143 valeurs extrêmes

2.  CORRELATIONS
    - has_loan ↔ emi_amount : 0.65 (logique)
    - is_fraud ↔ autres variables : 0.00-0.06 (faible)
    - Absence de redondance entre variables

3.  RISQUES PAR CANAL
    - API : risque critique (15-20% de fraude)
    - Web : risque élevé (5-8% de fraude)
    - Mobile_App : risque modéré (2-3% de fraude)
    - Canaux physiques : risque faible (< 1%)

4.  SAISONNALITE HORAIRE
    - Activité constante sur 24h
    - Données synthétiques
    - transaction_hour seule : pouvoir prédictif limité


DECISIONS DE MODELISATION
=========================

1.  CONSERVER LES OUTLIERS
    - Ce sont les anomalies que nous cherchons à détecter

2.  NORMALISER LES DONNEES
    - StandardScaler (problème d'échelle des montants)

3.  UTILISER ISOLATION FOREST
    - Les fraudes sont isolées dans l'espace

4.  UTILISER DBSCAN
    - Pour la segmentation et le clustering

5.  NE PAS UTILISER L'HEURE SEULE
    - Combiner avec d'autres variables (canal, montant)

6.  SURVEILLER L'API
    - Canal à haut risque : surveillance prioritaire


LIMITES DE L'ETUDE
==================

1.  DONNEES
    - Absence de données haïtiennes
    - Devise en INR (pas HTG)
    - Données synthétiques (pas de cycle humain)

2.  METHODOLOGIQUES
    - Besoin de surveillance continue en temps réel
    - Le modèle doit être ré-entraîné régulièrement

3.  STRATEGIES D'ATTENUATION
    - Ajuster les seuils pour le contexte haïtien
    - Ré-entraînement tous les 6 mois
    - Validation avec des données réelles


PROCHAINES ETAPES
=================

1.  Notebook 02 : Isolation Forest
    - Entraînement du modèle
    - Détection des anomalies
    - Visualisation des résultats

2.  Notebook 03 : DBSCAN + ACP
    - Entraînement de DBSCAN
    - Analyse ACP
    - Visualisation des clusters

3.  Notebook 04 : Evaluation finale
    - Comparaison des modèles
    - Métriques de performance
    - Rapport final


RESSOURCES
==========

- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

Installation des dépendances :
    pip install -r requirements.txt

