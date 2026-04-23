# Polymer Growth Simulation Software -- User Experience Evaluation

## Evaluator Information

| Field | Response |
|-------|----------|
| Name | |
| Role | [ ] Chemistry Researcher  [ ] Supervisor  [ ] Student  [ ] Other: _______ |
| Polymer science expertise | [ ] Beginner  [ ] Intermediate  [ ] Expert |
| Optimization software experience | [ ] None  [ ] Some  [ ] Extensive |
| Operating System | [ ] Windows  [ ] macOS  [ ] Linux |
| Date | |

---

## Instructions

This questionnaire evaluates the usability, functionality, and scientific accuracy of the Polymer Growth Simulation & Optimization software. It is structured using a 5-point Likert scale following standard HCI evaluation methodology.

**For each statement, circle the number that best reflects your experience:**

| 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| Strongly Disagree | Disagree | Neither Agree nor Disagree | Agree | Strongly Agree |

Where applicable, N/A may be used if a feature was not tested.

---

## Section 1: Installation and First Launch

| # | Statement | 1 | 2 | 3 | 4 | 5 | N/A |
|---|-----------|---|---|---|---|---|-----|
| 1.1 | The installation instructions were clear and complete. | | | | | | |
| 1.2 | The software installed without errors on my system. | | | | | | |
| 1.3 | I was able to launch the GUI within 5 minutes of starting installation. | | | | | | |
| 1.4 | The initial screen presented a clear starting point for new users. | | | | | | |

**Additional comments on installation experience:**

\
\

---

## Section 2: Simulation Tab -- Parameter Input

| # | Statement | 1 | 2 | 3 | 4 | 5 | N/A |
|---|-----------|---|---|---|---|---|-----|
| 2.1 | The parameter names are meaningful to someone with polymer chemistry knowledge. | | | | | | |
| 2.2 | The info icons (?) provide helpful explanations of each parameter. | | | | | | |
| 2.3 | The formulas shown in info tooltips are correct and clearly presented. | | | | | | |
| 2.4 | The default parameter values produce a plausible polymer distribution. | | | | | | |
| 2.5 | The input controls (spinboxes, checkboxes) have appropriate ranges and step sizes. | | | | | | |
| 2.6 | It is easy to understand which parameters have the largest effect on the output. | | | | | | |

**Suggested improvements for parameter input:**

\
\

---

## Section 3: Simulation Tab -- Results and Visualization

| # | Statement | 1 | 2 | 3 | 4 | 5 | N/A |
|---|-----------|---|---|---|---|---|-----|
| 3.1 | The chain length distribution plot is clear and informative. | | | | | | |
| 3.2 | The hover-to-inspect functionality on graphs is useful. | | | | | | |
| 3.3 | The zoom and pan controls help me examine specific regions of interest. | | | | | | |
| 3.4 | The "Save Plot" feature produces publication-quality figures. | | | | | | |
| 3.5 | The Mn, Mw, and PDI values are displayed clearly in the results panel. | | | | | | |
| 3.6 | The polymer metrics (Mn, Mw, PDI) appear to be calculated correctly. | | | | | | |
| 3.7 | The kinetics plot (Mn/Mw/PDI over time) provides useful reaction dynamics insight. | | | | | | |
| 3.8 | The chain populations plot (living/dead/conversion) is informative. | | | | | | |
| 3.9 | Exporting kinetics data to CSV/Excel is straightforward. | | | | | | |

**What additional visualization would be most valuable?**

\
\

---

## Section 4: Optimization Tab

| # | Statement | 1 | 2 | 3 | 4 | 5 | N/A |
|---|-----------|---|---|---|---|---|-----|
| 4.1 | Selecting and loading experimental data files is easy. | | | | | | |
| 4.2 | The FDDC algorithm configuration options are sufficient for my needs. | | | | | | |
| 4.3 | The CPU worker count selector is helpful for managing system resources. | | | | | | |
| 4.4 | Progress updates during optimization are frequent and informative enough. | | | | | | |
| 4.5 | The live convergence plot helps me understand optimization progress. | | | | | | |
| 4.6 | The console output provides useful diagnostic information. | | | | | | |
| 4.7 | The "Cancel" button works reliably to stop a running optimization. | | | | | | |
| 4.8 | The best parameters found are presented clearly and completely. | | | | | | |
| 4.9 | Automatic run saving to the runs/ folder is useful for record-keeping. | | | | | | |

**How long did a typical optimization take, and was it acceptable?**

\
\

---

## Section 5: Task Queue

| # | Statement | 1 | 2 | 3 | 4 | 5 | N/A |
|---|-----------|---|---|---|---|---|-----|
| 5.1 | Adding tasks to the queue is intuitive. | | | | | | |
| 5.2 | The batch run feature (varying seeds) is useful for statistical analysis. | | | | | | |
| 5.3 | The queue table clearly shows the status of each task. | | | | | | |
| 5.4 | Elapsed time and ETA estimates are accurate and helpful. | | | | | | |
| 5.5 | The live detail view provides enough information during long runs. | | | | | | |
| 5.6 | Queue management (remove, clear, cancel) works as expected. | | | | | | |

**Suggestions for improving the queue workflow:**

\
\

---

## Section 6: Scientific Accuracy and Validity

| # | Statement | 1 | 2 | 3 | 4 | 5 | N/A |
|---|-----------|---|---|---|---|---|-----|
| 6.1 | The simulation model captures the essential features of living polymerization. | | | | | | |
| 6.2 | The vampiric coupling mechanism is a reasonable model for combination termination. | | | | | | |
| 6.3 | The length-dependent coupling probability reflects known diffusion-controlled behavior. | | | | | | |
| 6.4 | The monomer depletion model is consistent with batch polymerization kinetics. | | | | | | |
| 6.5 | Optimization results produce distributions that fit experimental data well. | | | | | | |
| 6.6 | Results are reproducible when using the same random seed. | | | | | | |
| 6.7 | The cost function meaningfully distinguishes good fits from poor ones. | | | | | | |

**Scientific concerns or suggestions:**

\
\

---

## Section 7: Performance and Reliability

| # | Statement | 1 | 2 | 3 | 4 | 5 | N/A |
|---|-----------|---|---|---|---|---|-----|
| 7.1 | Single simulations complete in an acceptable time. | | | | | | |
| 7.2 | Optimization runs complete in an acceptable time for the quality of results. | | | | | | |
| 7.3 | Parallel evaluation provides a noticeable speedup over sequential execution. | | | | | | |
| 7.4 | The UI remains responsive (no freezing) during long-running operations. | | | | | | |
| 7.5 | The software did not crash during my testing session. | | | | | | |

**Performance issues encountered (include approximate run times):**

\
\

---

## Section 8: Overall Assessment

| # | Statement | 1 | 2 | 3 | 4 | 5 | N/A |
|---|-----------|---|---|---|---|---|-----|
| 8.1 | The software meets the needs of a polymer chemist for simulation work. | | | | | | |
| 8.2 | The software meets the needs of a polymer chemist for parameter fitting. | | | | | | |
| 8.3 | The software is an improvement over the previous (Thomas 2020) version. | | | | | | |
| 8.4 | I would recommend this software to a colleague working with polymer distributions. | | | | | | |
| 8.5 | The software is suitable for use in a research or educational setting. | | | | | | |

---

## Section 9: Open-Ended Feedback

### What aspects of the software did you find most valuable?

\
\
\

### What aspects of the software need the most improvement?

\
\
\

### Are there any features you expected but did not find?

\
\
\

### Any other comments, observations, or suggestions?

\
\
\

---

## Test Scenarios Completed

Please indicate which tasks you completed during your evaluation session:

- [ ] **A.** Ran a simulation with default parameters and examined the distribution plot
- [ ] **B.** Modified parameters (e.g., p_growth, p_death) and observed how the distribution changed
- [ ] **C.** Used the info icons to read parameter explanations
- [ ] **D.** Enabled kinetics tracking and examined the kinetics plot
- [ ] **E.** Exported kinetics data to CSV or Excel
- [ ] **F.** Used hover, zoom, and pan on a graph
- [ ] **G.** Saved a plot to PNG/SVG/PDF
- [ ] **H.** Loaded experimental data and ran an optimization
- [ ] **I.** Monitored optimization progress via convergence plot
- [ ] **J.** Added tasks to the queue and ran them
- [ ] **K.** Used the batch seed feature to queue multiple runs
- [ ] **L.** Compared results across different random seeds

---

*This questionnaire is part of the MSc thesis evaluation for the Polymer Growth Simulation & Optimization software (Basaran, 2026), building on the work of van den Broek (2020).*

*Evaluator signature: ________________________  Date: ____________*