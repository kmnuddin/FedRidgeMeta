import numpy as np

def entropy(p, axis=-1, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -(p*np.log(p)).sum(axis=axis)

def variation_ratio(p):
    return 1.0 - p.max(axis=1)

def top2_margin(p):
    K = p.shape[1]
    if K < 2:
        return np.zeros(p.shape[0])
    if K == 2:
        return np.abs(p[:, 0] - p[:, 1])
    part = np.partition(-p, 2, axis=1)
    top1 = -part[:, 0]
    top2 = -part[:, 1]
    return top1 - top2

def js_divergence(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    m = 0.5*(p+q)
    kl_pm = (p*np.log(p/m)).sum(axis=1)
    kl_qm = (q*np.log(q/m)).sum(axis=1)
    return 0.5*(kl_pm + kl_qm)

def summary_features_from_tree_probs(tree_probs, classes, feature_groups=None):
    """Build meta-features from per-tree probability predictions.

    Parameters
    ----------
    feature_groups : list[str] | None
        Which feature blocks to include.  Recognised groups:
            "mean_p"       – committee mean probabilities  (K cols)
            "var_p"        – per-class variance             (K cols)
            "disagreement" – JS-to-mean + variation ratio   (2 cols)
            "uncertainty"  – predictive entropy + top-2 margin (2 cols)
        None or ["all"] includes every group.
    """
    if feature_groups is None or feature_groups == ["all"] or "all" in feature_groups:
        feature_groups = ["mean_p", "var_p", "disagreement", "uncertainty"]

    P = np.stack(tree_probs, axis=0)
    mean_p = P.mean(axis=0)

    blocks = []
    names  = []

    if "mean_p" in feature_groups:
        blocks.append(mean_p)
        names += [f"mean_{c}" for c in classes]

    if "var_p" in feature_groups:
        var_p = P.var(axis=0)
        blocks.append(var_p)
        names += [f"var_{c}" for c in classes]

    if "disagreement" in feature_groups:
        js_each = np.stack([js_divergence(P[t], mean_p) for t in range(P.shape[0])], axis=0)
        js_to_mean = js_each.mean(axis=0)
        vr = variation_ratio(mean_p)
        blocks.append(js_to_mean[:, None])
        blocks.append(vr[:, None])
        names += ["js", "var_ratio"]

    if "uncertainty" in feature_groups:
        ent = entropy(mean_p)
        m2  = top2_margin(mean_p)
        blocks.append(ent[:, None])
        blocks.append(m2[:, None])
        names += ["entropy", "top2_margin"]

    X = np.concatenate(blocks, axis=1)
    return X, names


def summary_features_from_proba(proba, classes):
    """Build meta-features from a single model's probability output.

    Used for non-tree models (SVM, LR, MLP) that don't have per-estimator
    predictions. Produces mean_p + uncertainty features (K + 2 dims).

    Parameters
    ----------
    proba : ndarray (N, K)  — model's predict_proba output
    classes : list of K class labels

    Returns
    -------
    X : ndarray (N, K+2)
    names : list of str
    """
    ent = entropy(proba)
    m2  = top2_margin(proba)
    X = np.concatenate([proba, ent[:, None], m2[:, None]], axis=1)
    names = [f"mean_{c}" for c in classes] + ["entropy", "top2_margin"]
    return X, names

