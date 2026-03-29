from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def tfidf_lr(max_features: int = 50_000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            max_features=max_features, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)),
    ])


def tfidf_svm() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            max_features=30_000, sublinear_tf=True,
        )),
        ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)),
    ])


def knn_vote(k: int = 10) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("knn",   KNeighborsClassifier(n_neighbors=k, metric="cosine")),
    ])
