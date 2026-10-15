"""Tables and standalone plots from the verified component/encoding analysis."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
MODELS=["tabpfn","tabiclv2","xgboost","lightgbm","random_forest"]
NAMES={"tabpfn":"TabPFN","tabiclv2":"TabICLv2","xgboost":"XGBoost","lightgbm":"LightGBM","random_forest":"Random Forest"}
VARIANTS={"tabpfn":"outlier_off","tabiclv2":"outlier_off","xgboost":"exact","lightgbm":"fine_bins","random_forest":"leaf5"}
CHANGES={"tabpfn":"Outlier step off","tabiclv2":"Soft outlier step off","xgboost":"Exact split search","lightgbm":"1,023 bins","random_forest":"Minimum leaf size 5"}
CAT=[("xgboost","ordinal","legacy"),("xgboost","onehot","ordinal"),("xgboost","native","ordinal"),
     ("lightgbm","ordinal","legacy"),("lightgbm","onehot","ordinal"),("lightgbm","native","ordinal"),
     ("random_forest","ordinal","legacy"),("random_forest","onehot","ordinal"),("random_forest","onehot_full","ordinal_full")]
CAT_LABEL={"ordinal":"Ordinal minus legacy","onehot":"One-hot minus ordinal","native":"Native minus ordinal","onehot_full":"One-hot minus ordinal (all features)"}
METRICS=["roc_auc","balanced_accuracy","log_loss","ece_10_equal_width"]
METRIC_LABELS={"roc_auc":"ROC-AUC","balanced_accuracy":"Balanced accuracy","log_loss":"Log-loss reduction","ece_10_equal_width":"ECE reduction"}


def load(path):
    return pd.read_csv(path,float_precision="round_trip")


def match(frame,**where):
    selected=frame
    for key,value in where.items():
        selected=selected[selected[key]==value]
    if len(selected)!=1:
        raise ValueError(f"Expected one row for {where}, found {len(selected)}")
    return selected.iloc[0]


def cell(row,interval=True):
    if not np.isfinite(row["mean"]):
        return f"Undefined ({int(row.n_valid_datasets)}/{int(row.n_datasets)} complete datasets)"
    value=f"{row['mean']:+.4f}"
    return value+(f" [{row.ci_low:+.4f}, {row.ci_high:+.4f}]" if interval else "")


def markdown(path,headers,rows):
    lines=["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]
    lines.extend("| "+" | ".join(str(v) for v in row)+" |" for row in rows)
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def effect(summary,experiment,model,variant,base,rate,placement,estimand,metric="roc_auc",comparison=None):
    return match(summary,experiment=experiment,model=model,variant=variant,baseline=base,rate=rate,
        placement=placement,comparison=comparison or variant+"_vs_"+base,estimand=estimand,metric=metric)


def save(fig,folder,name):
    # Keep the declared 174-mm canvas so submission scaling preserves type size.
    fig.savefig(folder/(name+".pdf"))
    fig.savefig(folder/(name+".png"),dpi=300)
    plt.close(fig)


def generate(root,output):
    analysis=root/"analysis"
    if not analysis.exists() and (root/"reproduced").exists():
        analysis=root/"reproduced"
    verification=json.loads((analysis/"verification.json").read_text(encoding="utf-8"))
    if verification["status"]!="VERIFIED_FROM_PERSISTED_PREDICTIONS":
        raise ValueError("Verified complete reconstruction required")
    summary=load(analysis/"summary.csv")
    diag=load(analysis/"diagnostic_summary.csv")
    datasets=load(analysis/"paired_dataset_effects.csv")
    output.mkdir(parents=True,exist_ok=True)
    tables=output/"tables"
    figures=output/"figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    def values(experiment,model,variant,base,rate,placement,metric="roc_auc"):
        return [cell(effect(summary,experiment,model,variant,base,rate,placement,e,metric))
                for e in ["clean_gain","dirty_gain","degradation_reduction"]]
    rows=[[NAMES[m]+": "+CHANGES[m],*values("outlier",m,VARIANTS[m],"default",.5,"joint")] for m in MODELS]
    markdown(tables/"outlier_joint50.md",["Pipeline intervention","Clean change","Dirty change","Degradation reduction"],rows)
    rows=[[NAMES[m]+": "+CAT_LABEL[v],*values("category",m,v,b,.5,"joint")] for m,v,b in CAT]
    markdown(tables/"category_joint50.md",["Encoding contrast","Clean change","Dirty change","Degradation reduction"],rows)
    for experiment,models in [("outlier",[(m,VARIANTS[m],"default") for m in MODELS]),("category",CAT)]:
        rows=[]
        for rate in ([.05,.5] if experiment=="outlier" else [.15,.5]):
            for m,v,b in models:
                row=[f"{rate:.2f}",NAMES[m]+": "+(CHANGES[m] if experiment=="outlier" else CAT_LABEL[v])]
                row.extend(cell(effect(summary,experiment,m,v,b,rate,place,"dirty_gain")) for place in ["context","query","joint"])
                rows.append(row)
        markdown(tables/(experiment+"_placement_auc.md"),["Cell rate","Contrast","Context-only dirty change","Query-only dirty change","Joint dirty change"],rows)
        rows=[]
        for m,v,b in models:
            for metric in METRICS:
                rows.append([NAMES[m],METRIC_LABELS[metric],
                    CHANGES[m] if experiment=="outlier" else CAT_LABEL[v],*values(experiment,m,v,b,.5,"joint",metric)])
        markdown(tables/(experiment+"_secondary_joint50.md"),["Model","Oriented metric","Contrast","Clean change","Dirty change","Degradation reduction"],rows)
    failures=load(analysis/"numerical_failure_summary.csv")
    selected=failures[(failures.model.isin(["tabpfn","tabiclv2"]))&(failures.query_rate>0)]
    rows=[]
    for _,r in selected.iterrows():
        rows.append([NAMES[r.model],r.variant,f"{r.context_rate:.2f}",f"{r.query_rate:.2f}",
            f"{int(r.failed_views)}/{int(r.planned_views)}",int(r.datasets_with_any_failure),
            f"{r.dataset_mean_invalid_row_fraction:.2%}"])
    markdown(tables/"numerical_failures.md",["Model","Outlier setting","Context rate","Query rate","Failed full-query views","Datasets with failure","Mean invalid-row fraction"],rows)
    tree_rows=[]
    tree_metrics=["tree_mean_occupied_leaves","tree_mean_training_leaf_occupancy","tree_mean_fraction_leaves_at_most5_rows",
                  "tree_mean_max_depth","tree_fraction_numeric_splits_outside_reference_range"]
    for m in MODELS[2:]:
        for v in ["default",VARIANTS[m]]:
            for cr in [0.,.05,.5]:
                vals=[]
                for metric in tree_metrics:
                    r=match(diag,experiment="outlier",model=m,variant=v,context_rate=cr,query_rate=-1.,diagnostic=metric)
                    vals.append(f"{r['mean']:.4f}")
                tree_rows.append([NAMES[m],v,f"{cr:.2f}",*vals])
    markdown(tables/"tree_structures.md",["Model","Setting","Context rate","Occupied leaves","Training occupancy","Leaves with ≤5 rows","Maximum depth","Splits beyond clean range"],tree_rows)
    routing_rows=[]
    for m in MODELS[2:]:
        for v in ["default",VARIANTS[m]]:
            for rate in [.05,.5]:
                for cr in [0.,rate]:
                    vals=[]
                    for metric in ["routing_mean_tree_switch_fraction","routing_mean_switch_on_corrupted_rows","routing_mean_switch_on_unchanged_rows"]:
                        r=match(diag,experiment="outlier",model=m,variant=v,context_rate=cr,query_rate=rate,diagnostic=metric)
                        vals.append("NA" if not np.isfinite(r["mean"]) else f"{r['mean']:.4f}")
                    routing_rows.append([NAMES[m],v,f"{cr:.2f}",f"{rate:.2f}",*vals])
    markdown(tables/"fixed_tree_routing.md",["Model","Setting","Context rate","Query rate","Mean tree-switch fraction","Changed query rows","Unchanged query rows"],routing_rows)
    dimensions=load(analysis/"encoding_dimensions.csv")
    dim_rows=[]
    for dataset,group in dimensions[(dimensions.model=="random_forest")&(dimensions.variant=="onehot")].groupby("dataset_id"):
        clean=group[group.context_rate==0.]
        dirty=group[group.context_rate==.5]
        dim_rows.append([int(dataset),int(clean.source_features.iloc[0]),
            f"{int(clean.output_features.min())}–{int(clean.output_features.max())}",
            f"{int(dirty.output_features.min())}–{int(dirty.output_features.max())}",
            f"{int(clean.realized_features_per_split.min())}–{int(clean.realized_features_per_split.max())}"])
    markdown(tables/"encoding_dimensions.md",["OpenML ID","Original columns","One-hot columns: clean context","One-hot columns: 50% context faults","RF features per split: clean one-hot"],dim_rows)
    interaction=[]
    for rate in [.15,.5]:
        for place in ["context","query","joint"]:
            vals=[cell(effect(summary,"category","random_forest","onehot_full","ordinal_full",rate,place,e,
                  comparison="rf_encoding_effect_full_minus_sqrt")) for e in ["clean_gain","dirty_gain","degradation_reduction"]]
            interaction.append([f"{rate:.2f}",place,*vals])
    markdown(tables/"rf_feature_sampling_interaction.md",["Cell rate","Placement","Clean contrast","Dirty contrast","Degradation contrast"],interaction)

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.spines.top":False,"axes.spines.right":False,
        "pdf.fonttype":42,"ps.fonttype":42,"axes.labelsize":8.5,"axes.titlesize":9.5,
        "xtick.labelsize":8.5,"ytick.labelsize":8.5})
    fig,axes=plt.subplots(1,3,figsize=(174/25.4,4.2),sharey=True,layout="constrained")
    points=[]
    for ax,place,letter in zip(axes,["context","query","joint"],"abc"):
        ax.axvline(0,color="0.65",lw=.7)
        for i,m in enumerate(MODELS):
            for rate,offset,color,marker in [(.05,-.13,"#0072B2","o"),(.5,.13,"#D55E00","s")]:
                r=effect(summary,"outlier",m,VARIANTS[m],"default",rate,place,"dirty_gain")
                y=i+offset
                points.append(dict(model=m,rate=rate,placement=place,mean=r["mean"],ci_low=r.ci_low,ci_high=r.ci_high,
                                   n_valid_datasets=r.n_valid_datasets))
                if np.isfinite(r["mean"]):
                    ax.errorbar(r["mean"],y,xerr=[[r["mean"]-r.ci_low],[r.ci_high-r["mean"]]],fmt=marker,
                        color=color,ms=4,lw=.8,capsize=2,label=f"Cell rate {rate:.2f}" if i==0 else None)
                else:
                    ax.text(.97,y,f"Undefined ({int(r.n_valid_datasets)}/9)",ha="right",va="center",fontsize=8.5,
                            color=color,transform=ax.get_yaxis_transform())
        ax.set_title(f"({letter}) {place.capitalize()} only" if place!="joint" else f"({letter}) Joint")
        ax.set_xlabel("Dirty ROC-AUC change")
        ax.set_yticks(range(5),[NAMES[m]+"\n"+CHANGES[m] for m in MODELS])
        ax.grid(axis="x",alpha=.15)
    axes[0].invert_yaxis()
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,fontsize=8.5,frameon=False)
    save(fig,figures,"outlier_component_effects")
    pd.DataFrame(points).to_csv(figures/"outlier_component_effects.csv",index=False,float_format="%.17g")

    fig,axes=plt.subplots(1,2,figsize=(174/25.4,5.6),sharey=True,layout="constrained")
    category_points=[]
    for ax,place,letter in zip(axes,["query","joint"],"ab"):
        ax.axvline(0,color="0.65",lw=.7)
        for i,(m,v,b) in enumerate(CAT):
            for estimate,offset,color,marker in [("clean_gain",-.12,"#7F7F7F","o"),("dirty_gain",.12,"#0072B2","s")]:
                r=effect(summary,"category",m,v,b,.5,place,estimate)
                if not np.isfinite(r["mean"]):
                    raise ValueError("Category figure requires all seven datasets")
                ax.errorbar(r["mean"],i+offset,xerr=[[r["mean"]-r.ci_low],[r.ci_high-r["mean"]]],fmt=marker,
                    ms=4,color=color,lw=.8,capsize=2,label=estimate.replace("_"," ").capitalize() if i==0 else None)
                category_points.append(dict(model=m,variant=v,baseline=b,placement=place,estimand=estimate,
                    mean=r["mean"],ci_low=r.ci_low,ci_high=r.ci_high))
        for divider in [2.5,5.5]:
            ax.axhline(divider,color="0.85",lw=.7)
        ax.set_title(f"({letter}) "+("Query only" if place=="query" else "Joint"))
        ax.set_xlabel("ROC-AUC change")
        ax.set_yticks(range(len(CAT)),[NAMES[m]+":\n"+CAT_LABEL[v].replace(" (all features)","\n(all features)") for m,v,b in CAT])
        ax.grid(axis="x",alpha=.15)
    axes[0].invert_yaxis()
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,fontsize=8.5,frameon=False)
    save(fig,figures,"categorical_interface_effects")
    pd.DataFrame(category_points).to_csv(figures/"categorical_interface_effects.csv",index=False,float_format="%.17g")

    order=[3,24,31,802,42193,44158,44160]
    grid=np.empty((len(CAT),len(order)))
    for i,(m,v,b) in enumerate(CAT):
        for j,dataset in enumerate(order):
            r=match(datasets,experiment="category",dataset_id=dataset,model=m,variant=v,baseline=b,rate=.5,
                placement="joint",comparison=v+"_vs_"+b,estimand="dirty_gain",metric="roc_auc")
            grid[i,j]=r.value
    bound=float(np.max(np.abs(grid))) or .01
    fig,ax=plt.subplots(figsize=(174/25.4,5.3),layout="constrained")
    im=ax.imshow(grid,cmap="RdBu",vmin=-bound,vmax=bound,aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j,i,f"{grid[i,j]:+.3f}",ha="center",va="center",fontsize=8.5,
                    color="white" if abs(grid[i,j])>.6*bound else "black")
    ax.set_xticks(range(7),[str(v) for v in order])
    ax.set_xlabel("OpenML dataset ID")
    ax.set_yticks(range(len(CAT)),[NAMES[m]+":\n"+CAT_LABEL[v].replace(" (all features)","\n(all features)") for m,v,b in CAT])
    fig.colorbar(im,ax=ax,label="Dirty ROC-AUC change",shrink=.75)
    save(fig,figures,"categorical_dataset_effects")
    pd.DataFrame(grid,index=[m+":"+v+"-"+b for m,v,b in CAT],columns=order).to_csv(figures/"categorical_dataset_effects.csv",float_format="%.17g")
    report={"study_identity":verification["study_identity"],"fit_jobs":verification["fit_jobs"],
        "predictive_views":verification["predictive_views"],"numerical_failure_views":verification["numerical_failure_views"],
        "tables":sorted(p.name for p in tables.glob("*.md")),"figures":sorted(p.name for p in figures.glob("*.pdf"))}
    (output/"report_manifest.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=HERE)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    generate(args.root,args.output or args.root/"report")
