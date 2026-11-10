# Table: Performance Breakdown by Failure Type

## For inclusion in paper appendix or rebuttal

---

### Table: Pass@1 Performance by Failure Category (MiniMax-M2.5)

| Failure Category | Count | Baseline | +ReCAP | Gain |
|:-----------------|------:|---------:|-------:|-----:|
| Test Failure | 182 | 8.8 | 22.0 | +13.2 |
| Dependency Error | 135 | 5.2 | 16.3 | +11.1 |
| Code Formatting | 161 | 29.8 | 41.0 | +11.2 |
| Type Checking | 98 | 14.3 | 22.4 | +8.2 |
| Runtime Error | 78 | 12.8 | 20.5 | +7.7 |
| **Overall** | **408** | **19.6** | **31.9** | **+12.3** |
| **Excl. Format/Lint** | **305** | **13.8** | **24.6** | **+10.8** |

### Table: Pass@1 Performance by Failure Category (DeepSeek-V4-Flash)

| Failure Category | Count | Baseline | +ReCAP | Gain |
|:-----------------|------:|---------:|-------:|-----:|
| Test Failure | 182 | 11.0 | 21.4 | +10.4 |
| Dependency Error | 135 | 8.9 | 17.8 | +8.9 |
| Code Formatting | 161 | 34.2 | 42.9 | +8.7 |
| Runtime Error | 78 | 14.1 | 20.5 | +6.4 |
| Type Checking | 98 | 16.3 | 18.4 | +2.0 |
| **Overall** | **408** | **23.3** | **32.8** | **+9.6** |
| **Excl. Format/Lint** | **305** | **16.1** | **24.3** | **+8.2** |

---

## LaTeX Version

```latex
\begin{table}[t]
\centering
\caption{Pass@1 Performance Breakdown by Failure Category. Baseline is mini-SWE-agent without ReCAP experience. Instances may have multiple categories.}
\label{tab:failure_type_breakdown}
\small
\begin{tabular}{lrrrr|rrrr}
\toprule
\multirow{2}{*}{\textbf{Failure Category}} & \multicolumn{4}{c|}{\textbf{MiniMax-M2.5}} & \multicolumn{4}{c}{\textbf{DeepSeek-V4-Flash}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9}
& \textbf{n} & \textbf{Base} & \textbf{+ReCAP} & \textbf{Gain} & \textbf{n} & \textbf{Base} & \textbf{+ReCAP} & \textbf{Gain} \\
\midrule
Test Failure        & 182 & 8.8  & 22.0 & +13.2 & 182 & 11.0 & 21.4 & +10.4 \\
Dependency Error    & 135 & 5.2  & 16.3 & +11.1 & 135 & 8.9  & 17.8 & +8.9 \\
Code Formatting     & 161 & 29.8 & 41.0 & +11.2 & 161 & 34.2 & 42.9 & +8.7 \\
Type Checking       & 98  & 14.3 & 22.4 & +8.2  & 98  & 16.3 & 18.4 & +2.0 \\
Runtime Error       & 78  & 12.8 & 20.5 & +7.7  & 78  & 14.1 & 20.5 & +6.4 \\
\midrule
\textbf{Overall}    & \textbf{408} & \textbf{19.6} & \textbf{31.9} & \textbf{+12.3} & \textbf{408} & \textbf{23.3} & \textbf{32.8} & \textbf{+9.6} \\
\rowcolor{gray!20}
\textbf{Excl. Format/Lint} & \textbf{305} & \textbf{13.8} & \textbf{24.6} & \textbf{+10.8} & \textbf{305} & \textbf{16.1} & \textbf{24.3} & \textbf{+8.2} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Caption for the table

> **Table X: Pass@1 Performance Breakdown by Failure Category.** We report the top 5 most frequent failure categories in CI-Repair-Bench. Baseline is mini-SWE-agent without ReCAP experience. ReCAP achieves consistent gains across all major categories, including challenging categories like Test Failures (44.6% of instances) and Dependency Errors (33.1% of instances). The "Excl. Format/Lint" row shows performance on the 305 instances that have at least one non-formatting failure category. Gains persist strongly when excluding Code Formatting and Linting, demonstrating that ReCAP's value extends beyond simple style fixes. Some instances have multiple failure categories, so category counts sum to more than 408.

---

## Alternative: Vertical Layout

If space is tight, use this vertical layout:

### MiniMax-M2.5

| Category | n | Baseline | +ReCAP | Gain |
|:---------|--:|---------:|-------:|-----:|
| Test Failure | 182 | 8.8 | 22.0 | +13.2 |
| Dependency Error | 135 | 5.2 | 16.3 | +11.1 |
| Code Formatting | 161 | 29.8 | 41.0 | +11.2 |
| Type Checking | 98 | 14.3 | 22.4 | +8.2 |
| Runtime Error | 78 | 12.8 | 20.5 | +7.7 |
| **Overall** | **408** | **19.6** | **31.9** | **+12.3** |
| **Excl. Format/Lint** | **305** | **13.8** | **24.6** | **+10.8** |

### DeepSeek-V4-Flash

| Category | n | Baseline | +ReCAP | Gain |
|:---------|--:|---------:|-------:|-----:|
| Test Failure | 182 | 11.0 | 21.4 | +10.4 |
| Dependency Error | 135 | 8.9 | 17.8 | +8.9 |
| Code Formatting | 161 | 34.2 | 42.9 | +8.7 |
| Type Checking | 98 | 16.3 | 18.4 | +2.0 |
| Runtime Error | 78 | 14.1 | 20.5 | +6.4 |
| **Overall** | **408** | **23.3** | **32.8** | **+9.6** |
| **Excl. Format/Lint** | **305** | **16.1** | **24.3** | **+8.2** |
