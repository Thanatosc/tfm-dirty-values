"""Replot supplementary figures S4–S6 from the same recorded summaries."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from plot_previous import finish, select

HERE=Path(__file__).resolve().parent
MODEL_NAMES={"tabpfn":"TabPFN v2","tabiclv2":"TabICLv2","xgboost":"XGBoost",
             "lightgbm":"LightGBM","random_forest":"Random Forest"}
MECHANISMS=[("measurement_noise","Noise"),("outlier","Outlier"),("quantization","Quantization"),
            ("unit_error","Unit error"),("categorical_error","Categorical error")]
COLORS=["#0072B2","#D55E00","#009E73","#CC79A7","#222222"]


def panel(ax,letter):
    ax.spines[["top","right"]].set_visible(False)
    ax.text(0,1.03,letter,transform=ax.transAxes,fontsize=9,fontweight="bold")


def error_point(ax,row,y,**kwargs):
    ax.errorbar(row["mean"],y,xerr=[[row["mean"]-row.ci_low],[row.ci_high-row["mean"]]],capsize=2,lw=.9,ms=4,**kwargs)


def render_supplement(analysis,folder):
    analysis,folder=Path(analysis),Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"pdf.fonttype":42,"ps.fonttype":42})
    summary=pd.read_csv(analysis/"followup1/summary.csv",float_precision="round_trip")
    coverage=pd.read_csv(analysis.parent/"reference_analysis/diagnostics_original/row_coverage_macro.csv",float_precision="round_trip")
    cells=pd.read_csv(analysis.parent/"reference_analysis/diagnostics_original/cell_diagnostics_macro.csv",float_precision="round_trip")
    source_rows=[]
    diagnostic_rows=[]

    def result(experiment,model,estimand,corruption,rate):
        row=select(summary,experiment=experiment,model=model,estimand=estimand,corruption=corruption,rate=rate,metric="roc_auc")
        source_rows.append({"experiment":experiment,"model":model,"estimand":estimand,"corruption":corruption,
                            "rate":rate,"mean":float(row["mean"]),"ci_low":float(row.ci_low),"ci_high":float(row.ci_high)})
        return row

    fig,axes=plt.subplots(1,2,figsize=(174/25.4,4.25))
    for i,(mechanism,label) in enumerate(MECHANISMS):
        rows=coverage[(coverage.partition=="query")&(coverage.corruption==mechanism)].sort_values("rate")
        assert len(rows)==4
        diagnostic_rows.extend({"kind":"coverage","corruption":mechanism,"rate":float(r.rate),"cleaner":"none",
                                "value":float(r.selected_row_rate)} for r in rows.itertuples(index=False))
        axes[0].plot(rows.rate,rows.selected_row_rate,marker=["o","s","^","D","v"][i],ms=4,lw=1,color=COLORS[i],label=label)
    axes[0].set(xlabel="Selected-cell rate",ylabel="Fraction of query rows selected",ylim=(0,1.03))
    axes[0].legend(loc="lower right",fontsize=8.5,frameon=False)
    chosen=cells[(cells.partition=="query")&(cells.feature_type=="numeric")&((cells.rate==.15)|(cells.corruption=="clean"))&
                 (cells.corruption!="categorical_error")]
    order=["clean","measurement_noise","outlier","quantization","unit_error"]
    for delta,cleaner,color,label,hatch in [(-.18,"legacy_median_iqr","#D55E00","Median-IQR","//"),
                                           (.18,"canonical_tukey","#0072B2","Canonical Tukey","")]:
        values=[float(select(chosen,corruption=mechanism,cleaner=cleaner).collateral_rate)*100 for mechanism in order]
        diagnostic_rows.extend({"kind":"collateral_pct","corruption":mechanism,"rate":0.0 if mechanism=="clean" else .15,
                                "cleaner":cleaner,"value":value} for mechanism,value in zip(order,values))
        axes[1].bar(np.arange(5)+delta,values,width=.36,color=color,label=label,hatch=hatch,edgecolor="black",linewidth=.4)
    axes[1].set_xticks(range(5),["Clean","Noise","Outlier","Quant.","Unit"],rotation=35,ha="right")
    axes[1].set_ylabel("Clean numeric query cells changed (%)")
    axes[1].set_ylim(0,8.2)
    axes[1].legend(loc="upper left",fontsize=8.5,frameon=False)
    for i,ax in enumerate(axes):panel(ax,chr(97+i))
    fig.tight_layout(w_pad=1.5,pad=1.3)
    finish(fig,folder,"FigS4")

    fig=plt.figure(figsize=(174/25.4,6.5))
    grid=fig.add_gridspec(2,2)
    axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,:])]
    choices=[(-.13,"legacy_vs_uncleaned","#D55E00","s","Median-IQR"),
             (.13,"canonical_vs_uncleaned","#0072B2","o","Canonical Tukey")]
    for col,rate in enumerate([.15,.5]):
        ax=axes[col]
        for row,(mechanism,label) in enumerate(MECHANISMS):
            for offset,estimand,color,marker,legend in choices:
                estimate=result("canonical_tukey","tfm_mean",estimand,mechanism,rate)
                error_point(ax,estimate,row+offset,fmt=marker,color=color,label=legend if row==0 else None)
        ax.set_yticks(range(5),[name for _,name in MECHANISMS])
        ax.invert_yaxis();ax.set_xlabel("ROC-AUC versus uncleaned")
    ax=axes[2]
    for row,model in enumerate(MODEL_NAMES):
        for offset,estimand,color,marker,_ in choices:
            estimate=result("canonical_tukey",model,estimand,"clean",0)
            error_point(ax,estimate,row+offset,fmt=marker,color=color)
    ax.set_yticks(range(5),list(MODEL_NAMES.values()));ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC versus no cleaning")
    for i,ax in enumerate(axes):
        panel(ax,chr(97+i));ax.axvline(0,color="#555555",ls="--",lw=.7)
        ax.grid(axis="x",color="#e5e5e5",lw=.55);ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(3))
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.065,1,1),w_pad=1.5,h_pad=2.2,pad=1.4)
    finish(fig,folder,"FigS5")

    fig,axes=plt.subplots(3,2,figsize=(174/25.4,8.0),sharey=True)
    for col,(experiment,corruption) in enumerate([("robust_asinh","unit_error"),("category_permutation","categorical_error")]):
        for row,estimand in enumerate(["clean_change","dirty_change","degradation_reduction"]):
            ax=axes[row,col]
            for index,model in enumerate(MODEL_NAMES):
                estimate=result(experiment,model,estimand,corruption,.5)
                error_point(ax,estimate,index,fmt="o",color=COLORS[index])
            ax.set_yticks(range(5),list(MODEL_NAMES.values()))
            ax.axvline(0,color="#555555",ls="--",lw=.7)
            ax.grid(axis="x",color="#e5e5e5",lw=.55);ax.set_axisbelow(True)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.set_xlabel("ROC-AUC gap reduction" if estimand=="degradation_reduction" else "ROC-AUC change")
            panel(ax,chr(97+row*2+col))
    axes[0,0].invert_yaxis()
    fig.tight_layout(w_pad=1.5,h_pad=2.2,pad=1.4)
    finish(fig,folder,"FigS6")
    full_summary=pd.read_csv(analysis/"followup2/summary.csv",float_precision="round_trip")
    fig,axes=plt.subplots(2,3,figsize=(174/25.4,5.7))
    levels={"measurement_noise":[.1,.25,.5],"outlier":[3.,5.,10.],"quantization":[4.,8.,16.],
            "unit_error":[10.,100.,1000.],"categorical_error":[0.,.5,1.]}
    xlabels=["Noise SD /\ncontext SD","Distance in\ncontext SD","Equal-width bins","Multiplier magnitude","Novel-branch probability"]
    groups=[("tfm_mean","Two-TFM mean","#D55E00","o","-"),
            ("boosting_mean","Boosting mean","#0072B2","s","--"),
            ("random_forest","Random Forest","#555555","D",":")]
    parameter_rows=[];all_low=[];all_high=[]
    for index,(mechanism,grid) in enumerate(levels.items()):
        ax=axes.flat[index];panel(ax,chr(97+index))
        for model,label,color,marker,linestyle in groups:
            points=[select(full_summary,cohort="original",corruption=mechanism,level=level,rate=.5,
                           placement="joint",pattern="iid",representation="raw",contrast="clean_to_dirty",
                           estimand="degradation",metric="roc_auc",model=model) for level in grid]
            mean=np.array([r["mean"] for r in points]);low=np.array([r.ci_low for r in points]);high=np.array([r.ci_high for r in points])
            ax.errorbar(grid,mean,yerr=[mean-low,high-mean],marker=marker,linestyle=linestyle,color=color,
                        lw=.9,ms=3.5,capsize=2,label=label)
            parameter_rows.extend({"corruption":mechanism,"level":level,"model":model,"mean":float(r["mean"]),
                                   "ci_low":float(r.ci_low),"ci_high":float(r.ci_high)} for level,r in zip(grid,points))
            all_low.extend(low);all_high.extend(high)
        if mechanism in ["unit_error","quantization"]:ax.set_xscale("log",base=10 if mechanism=="unit_error" else 2)
        ax.set_xticks(grid,[f"{value:g}" for value in grid]);ax.minorticks_off()
        ax.set_xlabel(xlabels[index]);ax.axhline(0,color="#555555",lw=.7,ls="--")
        ax.grid(axis="y",color="#e5e5e5",lw=.55);ax.set_axisbelow(True)
    span=max(all_high)-min(0,min(all_low))
    for ax in list(axes.flat)[:5]:ax.set_ylim(min(0,min(all_low))-.04*span,max(all_high)+.05*span)
    axes[0,0].set_ylabel("ROC-AUC degradation");axes[1,0].set_ylabel("ROC-AUC degradation")
    axes[1,2].axis("off")
    handles,labels=axes[0,0].get_legend_handles_labels();axes[1,2].legend(handles,labels,loc="center",frameon=False)
    fig.tight_layout(w_pad=1.1,h_pad=2.0,pad=1.3)
    finish(fig,folder,"FigS7")
    pd.DataFrame(parameter_rows).to_csv(folder/"parameter_figure_values.csv",index=False,float_format="%.17g")
    pd.DataFrame(source_rows).to_csv(folder/"prediction_figure_values.csv",index=False,float_format="%.17g")
    pd.DataFrame(diagnostic_rows).to_csv(folder/"diagnostic_figure_values.csv",index=False,float_format="%.17g")
    (folder/"scope.json").write_text(json.dumps({"redrawn":["FigS4","FigS5","FigS6","FigS7"],"new_predictions":0,
        "predictive_values":len(source_rows),"predictive_source":"reproduced/followup1/summary.csv",
        "diagnostic_source":"reference_analysis/diagnostics_original/{row_coverage_macro,cell_diagnostics_macro}.csv",
        "changes":"Physical dimensions, font size, panel letters and non-colour encodings; all plotted estimands, conditions and values retained"},indent=2)+"\n",encoding="utf-8")
    from plot_retained import render_retained
    render_retained(analysis,folder)


if __name__=="__main__":
    render_supplement(HERE/"reproduced",HERE/"reproduced/figures/supplement")
