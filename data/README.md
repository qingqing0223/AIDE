# Data protocol

## AudioCaps
Expected local layout:
```text
data/raw/audiocaps/
  audio/
  train.csv
  val.csv
  test.csv
```
The original AudioCaps repository describes CSV fields as audiocap_id, youtube_id, start_time and caption. Raw audio is not committed here.

## VCTK 0.92
Expected local layout:
```text
data/raw/vctk/VCTK-Corpus-0.92/
```
Keep the microphone choice explicit. The KSEM run must record whether mic1 or mic2 is used.

## PSM prompt manifest
Required fields:
`id,text,label,is_harmful,source,split`
Labels are stable numeric IDs 0..6. The paper specifies one benign class and six harmful subsets; the retrieved source does not disclose the semantic names of those six subsets, so this rebuild uses `harmful_1` ... `harmful_6` and keeps source-to-class mapping external.

Raw prompt corpora are not copied into Git history.
