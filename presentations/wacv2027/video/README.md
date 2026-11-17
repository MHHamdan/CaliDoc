# CaliDoc: WACV 2027 talk videos

| File | Length | Format |
|---|---|---|
| `CaliDoc_WACV2027_Talk_full.mp4` | 9:41 | 1920×1080, H.264/AAC, soft English captions; within the 10-minute oral-video length used at WACV 2026 |
| `CaliDoc_WACV2027_Talk_short.mp4` | 4:07 | Same format; within the 5-minute poster-video length used at WACV 2026 |
| `*.srt` | — | Caption files (upload alongside the video on YouTube) |
| `*.chapters.txt` | — | YouTube chapter timestamps (paste into the description) |

**Narration** is an AI-generated voice (ElevenLabs, premade voice "Daniel") reading the authors' script. This is disclosed on the title frame and in the file metadata. The voice speaks about the authors in the third person and does not present itself as either author.

**Content:** the script (`../source/video_scripts.js`) uses only numbers from the accepted paper and supplement, audited in `../CaliDoc_WACV2027_Claim_Audit.md`. The narration was transcribed back with Whisper (small.en, with large-v3 spot checks), and every spoken number matches the script.

**Rebuild:** `ELEVENLABS_API_KEY=... python3 ../source/make_video.py full short`. Narration is cached per text hash, so unchanged slides cost nothing to rebuild.

## Suggested YouTube description

> CaliDoc: Calibrated Field-Level Risk and Annotation Auditing for Document Information Extraction. Mohammed Hamdan and Mohamed Cheriet, Synchromedia Laboratory, ÉTS Montréal. WACV 2027, Evaluations & Datasets Track, paper #1597.
>
> Token-level conformal coverage can look nominal while complete fields fail: on SROIE, 89.8% of tokens are covered but 0 of 420 fields. CaliDoc calibrates the complete field, bounding the expected fraction of missed fields per document, and uses the same threshold to audit annotations.
>
> Code: https://github.com/MHHamdan/CaliDoc · Supplement: https://github.com/MHHamdan/CaliDoc/releases/tag/wacv2027-camera-ready · Colab demo: https://colab.research.google.com/github/MHHamdan/CaliDoc/blob/wacv2027-demo-v1/notebooks/CaliDoc_WACV2027_Demo.ipynb
>
> Narration: AI-generated voice (ElevenLabs) reading the authors' script.
