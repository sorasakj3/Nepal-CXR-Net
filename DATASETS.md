# Datasets — Nepal-CXR-NET

This document gives an honest account of which datasets were actually used in training, which appear in the codebase as planned integrations, and what license governs each.

---

## Actually Used in Training

| Dataset | Images | Used For | License | Source |
|---------|--------|----------|---------|--------|
| TB Chest Radiography Database | ~4,200 | TB/Normal classifier training | CC BY 4.0 | Kaggle (Rahman et al., Qatar Univ. / Univ. of Dhaka) |
| Dataset of TB Chest X-rays | ~3,007 | TB/Normal classifier training | CC BY 4.0 | Kaggle |
| IQ-OTH/NCCD Lung Cancer Dataset | ~1,097 | Cancer class training | Research use | Kaggle (Iraqi cancer dataset — review license before redistribution) |
| NIH ChestX-ray14 | ~112,000 | YOLO pseudo-bbox generation only | NIH research use | NIH Clinical Center |

**CheXpert note:** CheXpert was present on disk but used for exactly one smoke test image to verify the inference pipeline loaded correctly. It was not used in any training run. License: Stanford research use only — do not redistribute.

---

## In Codebase But Never Used

The following datasets appear in `nepal_cxr_net/data/loaders/` as planned integrations. They were **never downloaded or incorporated into any training run** described in this project.

| Dataset | Loader File | Reason Not Used |
|---------|-------------|-----------------|
| VinDr-CXR | `vindr_cxr.py` | Never downloaded |
| TBX11K | `tbx11k.py` | Download scripts present; dataset never used |
| JSRT | `jsrt.py` | Never downloaded |
| MIMIC-CXR | `mimic_cxr.py` | Never downloaded (requires PhysioNet credentialing) |
| PadChest | `padchest.py` | Never downloaded |
| Shenzhen/Montgomery | `szch.py` | Never downloaded |

The loaders exist because they were written during architecture design as planned future integrations. Their presence in the codebase does not mean the datasets were used.

---

## License Notes for Demo / Display Use

If you plan to use any of these images in a public demo or video:

- **CC BY 4.0 datasets (TB Chest Radiography DB, Dataset of TB Chest X-rays):** Permitted for public display with attribution. Safe to use in a demo video.
- **NIH ChestX-ray14:** Research use license. Intended for "academic, research, and educational purposes." Public display in a demo is a gray area — when in doubt, use CC BY-licensed images instead.
- **CheXpert / MIMIC-CXR:** Strictly academic research licenses. Do not display these images publicly.
- **IQ-OTH/NCCD:** Kaggle research use. Verify license before public display.

**Recommendation:** For any public demo or video, use only images from the CC BY 4.0 TB datasets with proper attribution.

---

## Citation

If you use datasets from this project in your own work, please cite the original dataset papers rather than this repository. The dataset loaders include references to the original sources in their docstrings.
