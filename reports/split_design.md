# Split design

**Default dissertation split:** `B_chronological_fraction_70_15_15`

GoM ARGO export spans 2015–2022 only (not 2002–2020). Candidate A is empty. Chronological 70/15/15 preserves temporal order and balanced counts without hand-tuning year boundaries.

## Recommendations

1. **Default:** B_chronological_fraction_70_15_15
2. **High-observation test:** C_high_observation_test_2020
3. **Low-observation stress:** D_low_observation_stress_2015_2018
4. **Common-overlap era:** E_common_overlap_2015_2021
5. **Exclude:** 2022 (sparse tail, n=89)

## Candidate summaries

### A_simple_chronological_2002_2020
- **Counts:** {'train': 208, 'val': 332, 'test': 2415}
- **train years:** {'2015': 208}
- **val years:** {'2016': 135, '2017': 197}
- **test years:** {'2018': 151, '2019': 749, '2020': 1515}
- **Viable:** True

### E_common_overlap_2015_2021
- **Counts:** {'train': 1440, 'val': 1515, 'test': 1101}
- **train years:** {'2015': 208, '2016': 135, '2017': 197, '2018': 151, '2019': 749}
- **val years:** {'2020': 1515}
- **test years:** {'2021': 1101}
- **Viable:** True

### B_chronological_fraction_70_15_15
- **Counts:** {'train': 2901, 'val': 621, 'test': 623}
- **train years:** {'2015': 208, '2016': 135, '2017': 197, '2018': 151, '2019': 749, '2020': 1461}
- **val years:** {'2020': 54, '2021': 567}
- **test years:** {'2021': 534, '2022': 89}
- **Viable:** True

### C_high_observation_test_2020
- **Counts:** {'train': 1440, 'val': 1101, 'test': 1515}
- **train years:** {'2015': 208, '2016': 135, '2017': 197, '2018': 151, '2019': 749}
- **val years:** {'2021': 1101}
- **test years:** {'2020': 1515}
- **Viable:** True

### D_low_observation_stress_2015_2018
- **Counts:** {'train': 3365, 'val': 0, 'test': 691}
- **train years:** {'2019': 749, '2020': 1515, '2021': 1101}
- **val years:** {}
- **test years:** {'2015': 208, '2016': 135, '2017': 197, '2018': 151}
- **Viable:** True

