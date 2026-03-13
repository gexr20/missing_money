## Notes

- This repository contains analysis scripts for the Swiss electricity market **"missing money"** study.
- Input data are **Nexus-e simulation outputs** and are not included in this repository due to their large size.
- Scripts should be executed within each **scenario folder (outer folder)**.
- `Results` folders are excluded from version control.

---

## Code Structure

### Scenario: `run_nuclear_all_eu_evflex_8k_pathway_2050_DE_WRT_NF_SN`

**Direct data processing**
- `InUse_Average_comparison.py`
- `Final_Generation_cap_check_1st.py`
- `Final_Generation_cap_check_2nd.py`
- `aligned_NoOpCost_MarGen.py`
- `OpCost_MarGen_fast.py`

**Post-analysis**
- `Graph_OpCost_MarGen_fast.py`
- `post_analysis_saturation_1st.py`
- `post_analysis_saturation_2nd.py`
- `Post_analysis_marginal_gen.py`

---

### Scenario: `run_nuclear_all_eu_evflex_8k_pathway_2050_DE_NRT_NF_SN`

Contains the **same scripts as the WRT scenario**, with file names adjusted for the NRT case.

---

### Scenario: `tight_nuclear_all_eu_evflex_8k_pathway_2050_DE_WRT_NF_SN`

This scenario is used for **missing money calculation with a tighter solver configuration**.

**Main scripts**

- `tech_1st_sub_only_missing_money.py`

After running this script, place the following scripts inside the generated result folder and execute them sequentially:

1. `Sum_by_tech.py`
2. `plot_dualaxis.py`

For analysis with per MWh, run the following scripts inside the same result folder sequentially:

1. `perMWh_by_tech.py`
2. `perMWh_plot.py`

Additional analysis that considers different OPEX in 2 stages can be run with the following codes:

- `tech_missing_money_sanity_check.py`

After running this script, run the following scripts inside the same result folder sequentially:

1.  `perMWh_tech_aggregate.py`
2.  `plot.py`
