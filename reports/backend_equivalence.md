# gsw vs gsw_torch backend equivalence (F.3)

n_profiles=623  |  versions={'gsw': '3.6.19', 'gsw_torch': '0.1.1'}
**pass=False**  failed=['sigma0', 'spiciness0', 'SA', 'CT', 'N2']  N² flips @1e-8=14

| quantity | max_abs | RMS | atol | pass |
|----------|---------|-----|------|------|
| sigma0 | 9.03280956663366e-05 | 1.0524134163598021e-05 | 1e-06 | False |
| spiciness0 | 7.542735060978023e-05 | 1.0508698833953748e-05 | 1e-06 | False |
| SA | 0.00011424836903017876 | 1.341876015134547e-05 | 1e-06 | False |
| CT | 3.074025180183071e-06 | 2.3786928888018966e-07 | 1e-06 | False |
| N2 | 4.293949866854613e-08 | 3.4955629241303585e-09 | 1e-10 | False |

Headline metrics remain on reference gsw. Failures here are upstream gsw_torch discrepancy evidence (JOSS), not a STOP for evalphys.
