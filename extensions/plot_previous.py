"""Redraw the rev4 main figures from dataset estimates and complete summaries."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

MODELS = {"tabpfn": "TabPFN v2", "tabiclv2": "TabICLv2", "tfm_mean": "Two-TFM mean"}
COLORS = {"tabpfn": "#0072B2", "tabiclv2": "#D55E00", "tfm_mean": "#222222"}
MARKERS = {"tabpfn": "o", "tabiclv2": "s", "tfm_mean": "D"}
NAMES = {31:"credit-g",44:"spambase",802:"pbcseq",1049:"pc4",1487:"ozone-level-8hr",
         1504:"steel-plates-fault",42193:"compas-two-years",44158:"KDDCup09 upselling",44160:"rl",
         735:"cpu_small",752:"puma32H",833:"bank32nh",1479:"hill-valley",41156:"ada"}


def select(frame, **query):
    take = np.ones(len(frame), dtype=bool)
    for k, v in query.items():
        take &= frame[k].eq(v).to_numpy()
    rows = frame.loc[take]
    if len(rows) != 1:
        raise ValueError(f"Expected one result, got {len(rows)}: {query}")
    return rows.iloc[0]


def estimate(frame, **extra):
    query = dict(cohort="original", corruption="unit_error", level=1000., rate=.5,
                 placement="joint", pattern="iid", representation="asinh", contrast="raw_to_asinh",
                 estimand="dirty_change", metric="roc_auc", model="tfm_mean")
    return select(frame, **(query | extra))


def finish(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Locators may create ticks outside the view interval; Matplotlib does not
    # draw those ticks, even though their Text objects retain visible=True.
    undrawn_ticks=set()
    for ax in fig.axes:
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            low,high=sorted(limits)
            for tick in [*axis.get_major_ticks(),*axis.get_minor_ticks()]:
                if not low-1e-12 <= tick.get_loc() <= high+1e-12:
                    undrawn_ticks.update([tick.label1,tick.label2])
    outside=[]
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text().strip() or artist in undrawn_ticks:continue
        bounds=artist.get_window_extent(renderer)
        if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > fig.bbox.width+1 or bounds.y1 > fig.bbox.height+1:
            outside.append(artist.get_text())
    if outside:raise ValueError(f"Figure lettering outside canvas: {name}: {outside}")
    (folder/(name+"_layout_check.json")).write_text(json.dumps({"status":"PASS",
        "width_mm":float(fig.get_figwidth()*25.4),"height_mm":float(fig.get_figheight()*25.4),
        "outside_text":outside,"minimum_requested_font_pt":8.5},indent=2)+"\n",encoding="utf-8")
    for ext in ["pdf", "png", "eps"]:
        fig.savefig(folder / (name + "." + ext), dpi=600,
                    metadata={"Creator": "Matplotlib; recorded-metric reproduction"})
    plt.close(fig)


def axes_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color="#444444", linestyle="--", linewidth=.8)
    ax.grid(axis="x", color="#e5e5e5", linewidth=.55)
    ax.set_axisbelow(True)


def workflow(folder):
    # Physical dimensions and explicit line breaks keep the type readable at
    # the 174 mm submission width. Check the rendered text, not character counts.
    fig = plt.figure(figsize=(174 / 25.4, 94 / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 174), ylim=(0, 94)); ax.axis("off")
    checked = []

    def box(name, x, y, w, h, heading, body, color):
        patch = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=1.2",
                              facecolor=color,edgecolor="#555555",linewidth=.8)
        ax.add_patch(patch)
        title = ax.text(x+w/2, y+h-6, heading, ha="center", va="center",
                        fontsize=8.5, fontweight="bold", linespacing=1.25)
        detail = ax.text(x+w/2, y+(h-10)/2, body, ha="center", va="center",
                         fontsize=8.5, linespacing=1.5)
        checked.extend((name, kind, artist, (x+2,y+2,x+w-2,y+h-2))
                       for kind, artist in [("heading",title),("body",detail)])

    box("split",3,54,38,37,"Reference split",
        "70% context\n30% query\nFive paired seeds", "#E9F2F8")
    box("injection",46,54,38,37,"Encode / inject",
        "Fit on clean context\nInject into context,\nquery or both", "#E9F2F8")
    box("fitting",89,54,38,37,"Transform / clean",
        "Fit: supplied context\nRaw / affine / asinh\nor fixed cleaner", "#FFF0E0")
    box("prediction",132,54,38,37,"Fixed pipelines",
        "Two TFMs\nThree tree ensembles\nFour metrics", "#E9F2F8")
    box("diagnostics",29,3,116,31,"Offline cell diagnostics",
        "Known masks and pre-injection values; scoring only\n"
        "Detection, exact restoration, collateral edits\nand affected-row coverage", "#F3F3F3")
    for left,right in [(41,46),(84,89),(127,132)]:
        ax.add_patch(FancyArrowPatch((left+.4,72.5),(right-.4,72.5),
                                    arrowstyle="-|>",mutation_scale=8,linewidth=.8,
                                    shrinkA=0,shrinkB=0,color="#444444"))
    for x in [65,108]:
        ax.add_patch(FancyArrowPatch((x,53),(x,35),arrowstyle="-|>",
                                    mutation_scale=8,linewidth=.8,linestyle="--",
                                    shrinkA=0,shrinkB=0,color="#666666"))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    results=[]
    for name,kind,artist,bounds in checked:
        extent=artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        x0,y0,x1,y1=bounds
        passed=extent.x0>=x0 and extent.y0>=y0 and extent.x1<=x1 and extent.y1<=y1
        results.append({"box":name,"part":kind,"text":artist.get_text(),
                        "text_bounds_mm":list(extent.extents),"safe_bounds_mm":list(bounds),
                        "inside_with_2mm_padding":bool(passed)})
    if not all(r["inside_with_2mm_padding"] for r in results):
        raise ValueError("Workflow text overflow: "+json.dumps(results))
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"workflow_layout_check.json").write_text(json.dumps({
        "status":"PASS","figure_width_mm":174,"figure_height_mm":94,
        "font_size_pt":8.5,"checks":results},indent=2)+"\n",encoding="utf-8")
    for ext in ["pdf","png","eps"]:
        fig.savefig(folder/("workflow."+ext),dpi=600,
                    metadata={"Creator":"Matplotlib; recorded-study workflow"})
    plt.close(fig)


def scaling(summary, folder):
    names={"tabpfn":"TabPFN v2","tabiclv2":"TabICLv2","xgboost":"XGBoost","lightgbm":"LightGBM","random_forest":"Random Forest"}
    fig,axs=plt.subplots(2,2,figsize=(174/25.4,5.6),sharey=True)
    for row,(cohort,label,n) in enumerate([("original","Original",9),("external","Additional",5)]):
        for col,(outcome,title) in enumerate([("dirty_change","Dirty-input change"),("clean_change","Clean-input change")]):
            ax=axs[row,col]; axes_style(ax)
            for j,(contrast,rep,color,marker,name) in enumerate([
                ("raw_to_affine","affine","#0072B2","o","Raw to affine"),
                ("affine_to_asinh","asinh","#D55E00","s","Affine to asinh")]):
                points=[estimate(summary,cohort=cohort,model=m,estimand=outcome,contrast=contrast,representation=rep) for m in names]
                means=np.array([x["mean"] for x in points]);low=np.array([x.ci_low for x in points]);high=np.array([x.ci_high for x in points])
                ax.errorbar(means,np.arange(5)+(j-.5)*.24,xerr=[means-low,high-means],fmt=marker,color=color,ms=4,capsize=2,lw=1,label=name)
            ax.set_yticks(range(5),list(names.values()),fontsize=8.5)
            ax.text(0,1.025,chr(ord('a')+row*2+col),transform=ax.transAxes,fontsize=9,fontweight="bold")
            ax.xaxis.set_major_locator(MaxNLocator(3))
            if row==1: ax.set_xlabel("Change in ROC-AUC")
    axs[0,0].invert_yaxis()
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.065,1,1),w_pad=1.1,h_pad=1.8)
    finish(fig,folder,"scaling_decomposition")


def heterogeneity(dataset, folder):
    fig,axes=plt.subplots(2,1,figsize=(174/25.4,6.6),sharex=True,gridspec_kw={"height_ratios":[9,5]})
    extracted=[]
    for ax,(cohort,title,n) in zip(axes,[("original","Original panel",9),("external","Additional panel",5)]):
        selected=dataset[(dataset.cohort==cohort)&(dataset.corruption=="unit_error")&(dataset.level==1000.)&
                         (dataset.rate==.5)&(dataset.placement=="joint")&(dataset.pattern=="iid")&
                         (dataset.contrast=="raw_to_asinh")&(dataset.estimand=="dirty_change")&(dataset.metric=="roc_auc")]
        ids=sorted(selected.dataset_id.unique()); assert len(ids)==n
        axes_style(ax)
        for j,(model,label) in enumerate(MODELS.items()):
            values=[float(select(selected,dataset_id=d,model=model).value) for d in ids]
            ax.scatter(values,np.arange(n)+(j-1)*.21,c=COLORS[model],marker=MARKERS[model],s=27 if j<2 else 22,label=label,zorder=3)
            extracted.extend({"cohort":cohort,"dataset_id":int(d),"dataset":NAMES[int(d)],"model":model,"dirty_roc_auc_change":v} for d,v in zip(ids,values))
        ax.set_yticks(range(n),[f"{NAMES[int(d)]} ({d})" for d in ids],fontsize=8.5)
        ax.invert_yaxis()
        ax.text(0,1.03,"a" if cohort=="original" else "b",transform=ax.transAxes,fontsize=9,fontweight="bold")
        if cohort=="external":ax.set_xlabel("Raw-to-asinh dirty ROC-AUC change")
    values=np.array([r["dirty_roc_auc_change"] for r in extracted])
    margin=.07*(values.max()-values.min())
    axes[0].set_xlim(min(-.005,float(values.min()-margin)),float(values.max()+margin))
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False)
    fig.tight_layout(rect=(.025,.065,.995,1),h_pad=1.5,pad=1.4)
    finish(fig,folder,"dataset_heterogeneity")
    pd.DataFrame(extracted).to_csv(folder/"dataset_heterogeneity_data.csv",index=False,float_format="%.17g")


def boundaries(summary,folder):
    fig,axes=plt.subplots(1,2,figsize=(174/25.4,4.5),gridspec_kw={"width_ratios":[1.16,1]})
    ax=axes[0];axes_style(ax)
    scenarios=[("context","iid",.5,"Context only\ncell rate 0.50"),("query","iid",.5,"Query only\ncell rate 0.50"),
               ("joint","iid",.5,"Joint\ncell rate 0.50"),("query","row_batch",.15,"Query row cohort\nrow rate 0.15")]
    for j,(model,label) in enumerate(MODELS.items()):
        points=[estimate(summary,placement=p,pattern=pat,rate=r,model=model) for p,pat,r,_ in scenarios]
        m=np.array([x["mean"] for x in points]);lo=np.array([x.ci_low for x in points]);hi=np.array([x.ci_high for x in points])
        ax.errorbar(m,np.arange(4)+(j-1)*.20,xerr=[m-lo,hi-m],fmt=MARKERS[model],color=COLORS[model],ms=4,capsize=2,lw=.9,label=label)
    ax.set_yticks(range(4),[s[3] for s in scenarios],fontsize=8.5);ax.invert_yaxis()
    ax.text(0,1.025,"a",transform=ax.transAxes,fontsize=9,fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlabel("Dirty ROC-AUC change")
    ax=axes[1]
    ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",color="#e5e5e5",lw=.55);ax.set_axisbelow(True)
    for rate,color,marker in [(.15,"#009E73","o"),(.5,"#CC79A7","s")]:
        points=[estimate(summary,rate=rate,level=level) for level in [10.,100.,1000.]]
        m=np.array([x["mean"] for x in points]);lo=np.array([x.ci_low for x in points]);hi=np.array([x.ci_high for x in points])
        ax.errorbar(np.arange(3),m,yerr=[m-lo,hi-m],fmt=marker+"-",color=color,ms=4,capsize=2,lw=1,label=f"Cell rate {rate:.2f}")
    ax.axhline(0,color="#444444",linestyle="--",lw=.8);ax.set_xticks(range(3),["10","100","1000"])
    ax.set_xlabel("Unit multiplier\nor reciprocal");ax.set_ylabel("TFM dirty ROC-AUC change")
    ax.text(0,1.025,"b",transform=ax.transAxes,fontsize=9,fontweight="bold")
    ax.legend(loc="upper left",fontsize=8.5,frameon=False)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.10,1,1),w_pad=1.5)
    finish(fig,folder,"effect_boundaries")


def render(analysis,folder):
    analysis,folder=Path(analysis),Path(folder)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"pdf.fonttype":42,"ps.fonttype":42})
    summary=pd.read_csv(analysis/"followup2/summary.csv",float_precision="round_trip")
    dataset=pd.read_csv(analysis/"followup2/dataset_effects.csv",float_precision="round_trip")
    workflow(folder);scaling(summary,folder);heterogeneity(dataset,folder);boundaries(summary,folder)
    from plot_supplement_figures import render_supplement
    render_supplement(analysis,folder/"supplement")
    (folder/"figure_scope.json").write_text(json.dumps({"figures":4,"new_predictions":0,
        "input_summary":str((analysis/"followup2/summary.csv").name),
        "heterogeneity":"All 9 original and all 5 additional numeric datasets; five-seed means, no invented dataset-specific confidence intervals.",
        "intervals":"Existing descriptive, unadjusted dataset-bootstrap intervals.",
        "row_cohort":"Additional scenario differing in coverage and direction; not an isolated batch-structure effect."},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"FIGURES_RENDERED","figures":4}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--analysis-dir",default="reproduced");p.add_argument("--output-dir",default="reproduced/figures")
    a=p.parse_args();render(HERE/a.analysis_dir,HERE/a.output_dir)
