import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

def main():
    # 1. Load data
    data_path = "data/25x10_songs.json"
    with open(data_path, "r") as f:
        data = json.load(f)

    texts = [entry["lyrics"] for entry in data]
    labels = [entry["metadata"]["artist"] for entry in data]

    unique_artists = sorted(set(labels))
    n_classes = len(unique_artists)
    chance_level = 1.0 / n_classes
    print(f"Loaded {len(texts)} songs from {n_classes} artists.")

    # 2. Extract BoW features
    print("Extracting TF-IDF BoW features...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_sparse = vectorizer.fit_transform(texts)
    X_dense = X_sparse.toarray()

    # 3. PCA
    pca_components = 64
    print(f"Running PCA down to {pca_components} components...")
    pca = PCA(n_components=pca_components, random_state=42)
    X_pca = pca.fit_transform(X_dense)

    # 4. Probing configuration
    n_folds = 5
    max_iter = 300
    classifier = LogisticRegression(solver='lbfgs', max_iter=max_iter, random_state=42)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # 5. Cross validation
    print("Training linear probe (stratified K-fold)...")
    scores = cross_val_score(classifier, X_pca, labels, cv=cv, scoring='accuracy', n_jobs=-1)

    mean_acc = scores.mean()
    std_acc = scores.std()

    print(f"BoW Probe Accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")
    print(f"Chance Level: {chance_level:.3f}")

    # 6. Permutation test
    n_permutations = 1000
    print(f"Running permutation test with {n_permutations} permutations...")
    null_scores = []
    y_arr = np.array(labels)
    for i in range(n_permutations):
        rng = np.random.RandomState(42 + i)
        y_shuffled = rng.permutation(y_arr)
        perm_scores = cross_val_score(classifier, X_pca, y_shuffled, cv=cv, scoring='accuracy', n_jobs=-1)
        null_scores.append(perm_scores.mean())
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_permutations} permutations...")

    null_scores = np.array(null_scores)
    p_value = (np.sum(null_scores >= mean_acc) + 1.0) / (n_permutations + 1)
    
    print(f"Permutation p-value: {p_value:.4f}")
    print(f"Null mean: {null_scores.mean():.3f} +/- {null_scores.std():.3f}")

    # 7. Plotting
    plt.figure(figsize=(8, 5))
    sns.histplot(null_scores, bins=30, color="silver", stat="density", label=f"Null Distribution (n={n_permutations})", alpha=0.8)
    plt.axvline(mean_acc, color=sns.color_palette("muted")[0], linewidth=2.5, linestyle="-", label=f"Observed Accuracy ({mean_acc:.3f})")
    plt.axvline(chance_level, color="lightcoral", linestyle="--", linewidth=2, label=f"Chance ({chance_level:.3f})")

    plt.title(f"Permutation Null Distribution - BoW Baseline\np-value: {p_value:.4f} | PCA: {pca_components} | Folds: {n_folds}")
    plt.xlabel("Probe Accuracy")
    plt.ylabel("Density")
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig("bow_baseline_plot.png", dpi=150)
    print("Plot saved to bow_baseline_plot.png")

if __name__ == "__main__":
    main()
