# Figure package

Source for all plotted values: `results/full_study/analysis.json`  
Source SHA-256: `7d9389384762bc04bd352f0540f225d05d92b8f60b99d309232adc0d9ebbfe30`  
Transformation script: `scripts/generate_paper_results.py`  
Script SHA-256: `1d6f607156eabf9370b59869a8cfa4926fe4b1742319a8391df828cbfe9417bd`

## Figure 1

*ROC-AUC degradation across affected-cell rates and corruption operators*

Note. Points are model-specific dataset means and shaded bands are unadjusted 95% dataset-bootstrap intervals. Numeric corruptions use nine datasets; categorical error uses seven. Solid lines denote TFMs and dashed lines denote tree ensembles. The figure describes dose response; confirmatory H3 decisions use the preregistered exact tests and Holm correction.

Suggested placement: Results, RQ1/RQ3; double-column width.

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{paper/results/figures/figure_1_roc_auc_dose_response.pdf}
    \caption{ROC-AUC degradation across affected-cell rates and corruption operators. Points are model-specific dataset means; shaded bands are unadjusted 95\% dataset-bootstrap intervals.}
    \label{fig:dose-response}
\end{figure}
```

## Figure 2

*Preregistered confirmatory effect estimates for H1-H3*

Note. Points are dataset-level effect means and bars are unadjusted 95% dataset-bootstrap intervals. Filled points pass the preregistered Holm-adjusted exact-test threshold; open points do not. In H2, positive values mean greater degradation for TFMs. An H2 interval excluding zero is not treated as confirmatory significance when the exact Holm-adjusted p value is at least .05.

Suggested placement: Results, confirmatory tests; double-column width.

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{paper/results/figures/figure_2_confirmatory_effects.pdf}
    \caption{Preregistered confirmatory effect estimates for H1-H3. Bars show unadjusted 95\% dataset-bootstrap intervals; filled markers denote Holm-adjusted exact-test results below .05.}
    \label{fig:confirmatory-effects}
\end{figure}
```

## Visual verification

- Status: PASS
- Iterations: Figure 1 passed on first render; Figure 2 passed after one label-spacing correction
- Checks completed: source-series presence, interval scale, axis labels and units, legend readability, text clipping, color/shape distinction, and chart-junk review
- Remaining issues: none observed at the generated 300-DPI dimensions
- Data checks: exact plotted records are exported in `tables/table_s1_roc_auc_dose_response.csv` and `tables/table_1_confirmatory_hypotheses.csv`

## Figure/table trace

```yaml
figure_table_trace:
  - artifact_id: "fig-1"
    source_data:
      dataset_id: "dirty-values-full-study-v1"
      file: "results/full_study/analysis.json"
    transformation: {script: "scripts/generate_paper_results.py", hash: "1d6f607156eabf9370b59869a8cfa4926fe4b1742319a8391df828cbfe9417bd"}
    caption_claim: "ROC-AUC degradation generally increases as the affected-cell rate rises across the five corruption operators."
    supported_manuscript_claims:
      - claim: "ROC-AUC degradation generally increased as more eligible cells were affected, and all five H3 slopes were positive within the frozen panels."
        locator: "Section 4.2, Figure 1 and the two adjacent results paragraphs"
    limitations:
      - "Bands are unadjusted dataset-bootstrap intervals; confirmatory decisions come from separate exact tests with Holm correction."
      - "Numeric and categorical panels contain nine and seven datasets, respectively."
  - artifact_id: "fig-2"
    source_data:
      dataset_id: "dirty-values-full-study-v1"
      file: "results/full_study/analysis.json"
    transformation: {script: "scripts/generate_paper_results.py", hash: "1d6f607156eabf9370b59869a8cfa4926fe4b1742319a8391df828cbfe9417bd"}
    caption_claim: "H1 and H3 are supported for all corruption operators, whereas no H2 model-family contrast passes the frozen significance threshold."
    supported_manuscript_claims:
      - claim: "All five corruption operators significantly degrade TFMs and have positive degradation slopes within the frozen panels."
        locator: "Sections 4.2-4.3, Figure 2, and Table 3"
      - claim: "No TFM-versus-tree family contrast passes the preregistered alpha threshold."
        locator: "Section 4.4, Figure 2, and Table 3"
    limitations:
      - "The confidence intervals are unadjusted and are not the decision rule for confirmatory significance."
      - "Non-significant H2 results do not establish equivalence."
  - artifact_id: "table-1-confirmatory-hypotheses"
    source_data:
      dataset_id: "dirty-values-full-study-v1"
      file: "results/full_study/analysis.json"
    transformation: {script: "scripts/generate_paper_results.py", hash: "1d6f607156eabf9370b59869a8cfa4926fe4b1742319a8391df828cbfe9417bd"}
    caption_claim: "The table reports every preregistered H1-H3 result without outcome-based selection."
    supported_manuscript_claims:
      - claim: "H1 and H3 are supported for all five corruptions; H2 detects no family contrast after correction."
        locator: "Sections 4.2-4.4 and manuscript Table 3"
    limitations:
      - "Exact tests have coarse resolution because the independent-unit count is seven or nine datasets."
  - artifact_id: "table-2-rate-050-by-model"
    source_data:
      dataset_id: "dirty-values-full-study-v1"
      file: "results/full_study/analysis.json"
    transformation: {script: "scripts/generate_paper_results.py", hash: "1d6f607156eabf9370b59869a8cfa4926fe4b1742319a8391df828cbfe9417bd"}
    caption_claim: "The table reports model-specific ROC-AUC degradation at the preregistered primary rate."
    supported_manuscript_claims:
      - claim: "ROC-AUC degradation at rate 0.50 is positive across all model-corruption combinations."
        locator: "Section 4.4, final paragraph and linked rate-0.50 results table"
    limitations:
      - "Model-specific intervals are descriptive and not additional confirmatory tests."
  - artifact_id: "table-3-rq4-summary"
    source_data:
      dataset_id: "dirty-values-full-study-v1"
      file: "results/full_study/analysis.json"
    transformation: {script: "scripts/generate_paper_results.py", hash: "1d6f607156eabf9370b59869a8cfa4926fe4b1742319a8391df828cbfe9417bd"}
    caption_claim: "Balanced accuracy and log loss broadly track ROC-AUC degradation, while ECE is less consistent."
    supported_manuscript_claims:
      - claim: "There is no descriptive evidence that ECE generally deteriorates before ROC-AUC."
        locator: "Section 4.5, manuscript Table 4 and onset-diagnostic paragraph"
    limitations:
      - "RQ4 and onset comparisons are descriptive and are not multiple-comparison-adjusted confirmatory tests."
```
