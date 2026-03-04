import numpy as np

def entropy(p, axis=-1, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -(p*np.log(p)).sum(axis=axis)

def variation_ratio(p):
    return 1.0 - p.max(axis=1)

def top2_margin(p):
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

def summary_features_from_tree_probs(tree_probs, classes):
    P = np.stack(tree_probs, axis=0)  # [T, N, C]
    mean_p = P.mean(axis=0)
    var_p  = P.var(axis=0)
    js_each = np.stack([js_divergence(P[t], mean_p) for t in range(P.shape[0])], axis=0)
    js_to_mean = js_each.mean(axis=0)
    vr = variation_ratio(mean_p)
    ent = entropy(mean_p)
    m2 = top2_margin(mean_p)
    X = np.concatenate([mean_p, var_p, js_to_mean[:,None], vr[:,None], ent[:,None], m2[:,None]], axis=1)
    names = [f"mean_{c}" for c in classes] + [f"var_{c}" for c in classes] + ["js", "var_ratio", "entropy", "top2_margin"]
    return X, names
