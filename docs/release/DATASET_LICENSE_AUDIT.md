# License audit: code, weights, datasets

Checked 2026-10-17 against primary sources. Quotes are verbatim from the URLs given.
When a license is only asserted by a mirror and not by the dataset authors, the
row is marked UNCLEAR.

## Three separate licenses

| artifact | license | basis |
|---|---|---|
| CaliDoc source code (this repository) | Apache-2.0 | `setup.py` classifier, chosen by the author |
| Released model weights (`MHamdan/calidoc-layoutlmv3-large-*`) | **CC BY-NC-SA 4.0** | inherited from the base model (ShareAlike); see below |
| Datasets | per dataset, see table | not redistributed except the CORD showcase examples |

### Base model

- `microsoft/layoutlmv3-large`, https://huggingface.co/microsoft/layoutlmv3-large/raw/main/README.md:
  the YAML gives `license: cc-by-nc-sa-4.0`, and the text reads "The content of this project
  itself is licensed under the Attribution-NonCommercial-ShareAlike 4.0 International
  (CC BY-NC-SA 4.0)".
- `microsoft/layoutlmv3-base` (whose processor files are shipped) carries the same license.
- The unilm LayoutLMv3 page (https://github.com/microsoft/unilm/tree/master/layoutlmv3)
  carries the same statement.

Fine-tuned weights are adaptations of the base model, so they are released under CC BY-NC-SA
4.0, with attribution to Huang et al. 2022 and a statement of the modification (token-
classification fine-tuning plus calibration files). The weights are **not** released under
the code's Apache-2.0 license.

## Datasets

| dataset | official source | lineage used by CaliDoc | license | images redistributable | annotations redistributable | attribution | notes |
|---|---|---|---|---|---|---|---|
| CORD v2 | https://github.com/clovaai/cord, https://huggingface.co/datasets/naver-clova-ix/cord-v2 | HF `naver-clova-ix/cord-v2` @ `7f0115a4b758a71d6473b8d085751692da2fef98` | CC BY 4.0, stated by the authors | **YES** (with attribution) | **YES** (with attribution) | Park et al., "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing", 2019, plus a license link | The GitHub license file `LICENSE-CC-BY` has SPDX `CC-BY-4.0`. README: "This work is licensed under a Creative Commons Attribution 4.0 International License." The HF card has `license: cc-by-4.0`. Personal information in the images is blurred by the dataset authors |
| FUNSD | https://guillaumejaume.github.io/FUNSD/ (terms at /FUNSD/work/) | official `dataset.zip`, sha256 `c3173564…761f` | FUNSD License (EPFL-LTS5): non-commercial research and education only | **NO** | **UNCLEAR** (no grant to redistribute) | Jaume, Ekenel, Thiran, ICDAR-OST 2019 | "Use of the FUNSD Dataset is solely for non-commercial, research and educational purposes." "Licensee's use of the images is governed by a copyright. … Original images are part of the dataset RVL-CDIP." The HF mirror `nielsr/funsd` declares no license |
| WildReceipt | Sun et al. 2021 (arXiv:2103.14470); canonical host `download.openmmlab.com/mmocr/data/wildreceipt.tar` (unreachable at retrieval) | two byte-identical Kaggle mirrors (`shregar/doc-wildreceipt`, `maazahmad69/wild-receipt`) | **none stated by the authors** (MMOCR `dataset_zoo/wildreceipt/metafile.yml`: `License: Type: N/A`) | **UNCLEAR, treated as NO** | **UNCLEAR** | Sun et al. 2021 | The Kaggle license fields ("Other", "Apache 2.0") are mirror assertions with no upstream basis |
| SROIE | ICDAR 2019 RRC, https://rrc.cvc.uab.es/?ch=13 (login-gated downloads) | `darentang/sroie` pre-tagged zip, sha256 `8606210316…1fc4` | CC BY 4.0 **asserted only by third parties** (e.g. `jsdnrs/ICDAR2019-SROIE` card, MMOCR metafile); no license text on the public RRC pages; `darentang/sroie` declares none | **UNCLEAR, treated as NO** | **UNCLEAR** | Huang et al., ICDAR 2019 SROIE competition (arXiv:2103.10213) | Could be confirmed by logging in to RRC or contacting the organizers |

## Consequences for the release

- **README and assets.** Only CORD images, OCR words, boxes and gold annotations appear in
  `assets/examples/`, with the attribution above. Each showcase figure carries the credit
  line.
- **FUNSD, WildReceipt and SROIE examples.** `assets/examples/<ds>/<doc_id>/prediction.json`
  keeps document IDs, field types, word positions, set sizes, thresholds and statuses. It
  contains no words, boxes, images or gold labels. `tools/showcase_examples.py --full`
  regenerates the complete output on a local copy of the dataset.
- **Weight repositories.** These contain no dataset images or annotations. The calibration
  manifests list document identifiers only.
- **Training-data terms on the model cards.** Each card states its dataset's terms. FUNSD
  is non-commercial, which is consistent with the NC weight license. WildReceipt has no
  stated license and SROIE has an unconfirmed one: the cards say so and point users to the
  official sources.
