"""Pinned estimator adapters and observational component diagnostics.

No installed package files are edited. Component switches use constructors.
The TabICL wrappers only record transformations and return the original output.
"""
from __future__ import annotations
from contextlib import ExitStack
import json
import numpy as np
from unittest.mock import patch
from core import previous, mean_or_none, array_sha
from tfm_dirty_values.config import resolve_path
from tfm_dirty_values.models.adapters import _tabicl_empty_feature_compatibility

class FittedPipeline:
    def __init__(self, job, cats, x, y):
        self.job, self.cats = job, list(cats)
        self.trace = []
        self.stage = "fit_context"
        self.stack = ExitStack()
        self.imputer = None
        self.empty_indices = []
        self.metadata = {}
        cfg = previous.config()
        settings = cfg["models"][job["model"]]
        seed, variant, model = job["seed"], job["variant"], job["model"]
        n = int(settings["n_estimators"])
        self.metadata.update(package=settings["package"], requested_estimators=n, seed=seed)
        try:
            if model == "xgboost":
                from xgboost import XGBClassifier
                native = job["experiment"] == "category" and variant == "native"
                options = dict(n_estimators=n, tree_method="exact" if variant == "exact" else "hist",
                    n_jobs=4, random_state=seed, eval_metric="logloss")
                if native:
                    options.update(enable_categorical=True, feature_types=["c" if c in cats else "q" for c in range(x.shape[1])])
                self.model = XGBClassifier(**options)
                self.model.fit(x, y)
                self.metadata.update(tree_method=options["tree_method"], native_categories=native,
                    max_bin=int(self.model.get_xgb_params().get("max_bin") or 256))
                if native:
                    actual = self.model.get_booster().feature_types
                    if actual != options["feature_types"]:
                        raise ValueError("XGBoost category flags did not reach the fitted booster")
                    self.metadata["realized_feature_types"] = actual
            elif model == "lightgbm":
                from lightgbm import LGBMClassifier
                native = job["experiment"] == "category" and variant == "native"
                options = dict(n_estimators=n, n_jobs=4, random_state=seed, verbosity=-1)
                if variant == "fine_bins":
                    options["max_bin"] = 1023
                self.model = LGBMClassifier(**options)
                self.model.fit(x, y, **({"categorical_feature":cats} if native else {}))
                self.metadata.update(native_categories=native, max_bin=options.get("max_bin", 255),
                    actual_trees=int(self.model.booster_.num_trees()))
                if native:
                    actual = self.model.booster_.params.get("categorical_column", [])
                    if list(actual) != cats:
                        raise ValueError(f"LightGBM category flags differ: {actual}")
                    self.metadata["realized_categorical_columns"] = list(actual)
            elif model == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.impute import SimpleImputer
                from joblib import parallel_backend
                self.stack.enter_context(parallel_backend("sequential"))
                self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
                x = self.imputer.fit_transform(x)
                self.model = RandomForestClassifier(n_estimators=n, n_jobs=4, random_state=seed,
                    min_samples_leaf=5 if variant == "leaf5" else 1,
                    max_features=1.0 if variant.endswith("_full") else "sqrt")
                self.model.fit(x, y)
                self.metadata.update(min_samples_leaf=self.model.min_samples_leaf,
                    max_features=self.model.max_features, realized_features_per_split=int(self.model.estimators_[0].max_features_),
                    backend="joblib_sequential", actual_trees=len(self.model.estimators_))
            elif model == "tabpfn":
                from tabpfn import TabPFNClassifier
                options = dict(n_estimators=n, categorical_features_indices=cats or None,
                    model_path=resolve_path(cfg, settings["checkpoint"]), device=settings["device"],
                    random_state=seed, fit_mode="low_memory", n_jobs=1)
                if variant == "outlier_off":
                    options["inference_config"] = {"OUTLIER_REMOVAL_STD":None}
                self.model = TabPFNClassifier(**options)
                self.model.fit(x, y)
                layers = [m for m in self.model.model_.modules() if hasattr(m, "remove_outliers")]
                active = [bool(m.remove_outliers) for m in layers]
                if not active or any(flag != (variant != "outlier_off") for flag in active):
                    raise ValueError("TabPFN realized outlier switch differs")
                self.metadata.update(outlier_active=active,
                    outlier_sigmas=[float(m.remove_outliers_sigma) if getattr(m,"remove_outliers",False) else None for m in layers],
                    checkpoint_sha256=settings["checkpoint_sha256"], fit_mode="low_memory")
            elif model == "tabiclv2":
                from tabicl import TabICLClassifier
                from tabicl._sklearn.preprocessing import OutlierRemover, CustomStandardScaler
                self._observe(OutlierRemover, "soft_outlier")
                self._observe(CustomStandardScaler, "standardization")
                self.empty_indices = np.flatnonzero(np.isnan(x).all(axis=0)).tolist()
                x, _, compatibility = _tabicl_empty_feature_compatibility(x, x)
                threshold = float("inf") if variant == "outlier_off" else 4.0
                self.model = TabICLClassifier(n_estimators=n, batch_size=int(settings["batch_size"]),
                    model_path=resolve_path(cfg,settings["checkpoint"]), allow_auto_download=False,
                    checkpoint_version=settings["checkpoint_version"], device=settings["device"],
                    random_state=seed, verbose=False, outlier_threshold=threshold)
                self.model.fit(x, y)
                if self.model.outlier_threshold != threshold:
                    raise ValueError("TabICL outlier threshold differs")
                self.metadata.update(outlier_threshold="infinity" if variant=="outlier_off" else 4.0,
                    standardization_and_100_clip_retained=True, empty_feature_compatibility=compatibility,
                    checkpoint_sha256=settings["checkpoint_sha256"])
            else:
                raise ValueError(model)
            self.X_fit = x
            self.metadata["fitted_input_sha256"] = array_sha(x)
            self.metadata["n_features"] = int(x.shape[1])
        except BaseException:
            self.stack.close()
            raise

    def _observe(self, cls, label):
        original = cls.transform
        owner = self
        def observe(transformer, x, *args, **kwargs):
            output = original(transformer, x, *args, **kwargs)
            before = np.asarray(x)
            after = np.asarray(output)
            entry = dict(stage=owner.stage, component=label, cells=int(before.size),
                changed_cells=int(np.count_nonzero(before != after)))
            if label == "soft_outlier":
                entry["threshold"] = "infinity" if np.isinf(transformer.threshold) else float(transformer.threshold)
                if owner.job["variant"]=="outlier_off" and not np.array_equal(before, after):
                    raise ValueError("Infinite-threshold soft outlier step changed an input")
            else:
                unbounded = (before-transformer.mean_) / transformer.scale_
                entry["hard_clip_cells"] = int(np.count_nonzero(unbounded != after))
            owner.trace.append(entry)
            return output
        self.stack.enter_context(patch.object(cls, "transform", observe))

    def prepare_query(self, x):
        if self.empty_indices:
            x = x.copy()
            x[:, self.empty_indices] = 0.0
        if self.imputer is not None:
            x = self.imputer.transform(x)
        return x

    def predict(self, x, stage):
        self.stage = stage
        x = self.prepare_query(x)
        p = np.asarray(self.model.predict_proba(x))[:, 1].astype(np.float64)
        if p.ndim != 1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError("Invalid predicted probabilities")
        return p

    def leaves(self, x, fitted=False):
        if not fitted:
            x = self.prepare_query(x)
        if self.job["model"]=="lightgbm":
            leaves = self.model.booster_.predict(x, pred_leaf=True)
        else:
            leaves = self.model.apply(x)
        return np.asarray(leaves, dtype=np.int32).reshape(len(x), -1)

    def tree_diagnostics(self, clean_context):
        leaves = self.leaves(self.X_fit, fitted=True)
        counts = [np.unique(leaves[:,i], return_counts=True)[1] for i in range(leaves.shape[1])]
        result = dict(actual_trees=int(leaves.shape[1]),
            mean_occupied_leaves=float(np.mean([len(c) for c in counts])),
            mean_fraction_leaves_at_most5_rows=float(np.mean([np.mean(c<=5) for c in counts])),
            mean_training_leaf_occupancy=float(np.mean([np.mean(c) for c in counts])))
        splits, depths = [], []
        model = self.job["model"]
        if model=="random_forest":
            for est in self.model.estimators_:
                t = est.tree_
                splits.extend((int(f),float(z)) for f,z in zip(t.feature,t.threshold) if f>=0)
                depths.append(int(t.max_depth))
        elif model=="xgboost":
            def visit(node, depth):
                if "children" not in node:
                    return depth
                feature = int(str(node["split"]).removeprefix("f"))
                splits.append((feature,float(node["split_condition"])))
                return max(visit(n,depth+1) for n in node["children"])
            depths = [visit(json.loads(tree),0) for tree in self.model.get_booster().get_dump(dump_format="json")]
        elif model=="lightgbm":
            def visit(node, depth):
                if "split_index" not in node:
                    return depth
                splits.append((int(node["split_feature"]),float(node["threshold"])))
                return max(visit(node["left_child"],depth+1),visit(node["right_child"],depth+1))
            depths = [visit(t["tree_structure"],0) for t in self.model.booster_.dump_model()["tree_info"]]
        numeric = [(f,z) for f,z in splits if f not in self.cats]
        bounds = {}
        for f in {f for f,_ in numeric}:
            v = clean_context[np.isfinite(clean_context[:,f]),f]
            bounds[f] = None if not len(v) else (float(v.min()),float(v.max()),float(v.mean()),float(v.std()))
        usable = [(f,z) for f,z in numeric if bounds[f] is not None]
        outside = [z<bounds[f][0] or z>bounds[f][1] for f,z in usable]
        distant = [abs(z-bounds[f][2])>3*max(bounds[f][3],1e-12) for f,z in usable]
        result.update(total_splits=len(splits), numeric_splits=len(numeric),
            mean_max_depth=mean_or_none(depths),
            fraction_numeric_splits_outside_reference_range=mean_or_none(outside),
            fraction_numeric_splits_beyond_reference3sd=mean_or_none(distant),
            bounds_source="clean_context_offline_diagnostic_only")
        return result

    def close(self):
        self.stack.close()
        self.model = None
        self.X_fit = None

def routing_summary(clean_leaves, dirty_leaves, changed_rows):
    if clean_leaves.shape != dirty_leaves.shape:
        raise ValueError("Leaf comparison must use the same fitted ensemble")
    changed = clean_leaves != dirty_leaves
    per_row = np.mean(changed, axis=1)
    result = dict(mean_tree_switch_fraction=float(np.mean(per_row)),
        fraction_rows_any_leaf_switch=float(np.mean(per_row>0)),
        mean_switch_on_corrupted_rows=mean_or_none(per_row[changed_rows]),
        mean_switch_on_unchanged_rows=mean_or_none(per_row[~changed_rows]),
        corrupted_rows=int(changed_rows.sum()), query_rows=len(changed_rows),
        comparison="clean_and_dirty_queries_in_one_fixed_fitted_model")
    if np.any(per_row[~changed_rows] != 0):
        raise ValueError("A fixed tree changed routing for an unchanged query")
    return result, per_row
