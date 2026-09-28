# Forensic Model Audit Report

## Audit scope

This audit covers only the legacy project located at:

`D:\E_Drive\Plant_disease\Plant-Disease-Detection-and-Classification-System\Plant-Disease-Detection-main`

The literal path written in the request as `D:\E\_Drive\...` does not exist in
this environment. The matching nested project above was audited. The parent
PlantCare project and its model were not used.

Audit date: 2026-09-27.

No production source, model, dataset, dependency file, or application behavior
was changed. All generated artifacts are under `_audit/`.

## 1. Executive summary

The saved model is a valid PyTorch state dictionary and loads strictly into the
39-class CNN defined in `Flask Deployed App/CNN.py`. It has 52,595,399
parameters, accepts `1 x 3 x 224 x 224` tensors, produces `1 x 39` finite
logits, and can run inference.

The reported notebook result is not independently reproducible as a final model
claim. The notebook contains a random split without a seed, does not save split
membership, records no epoch-level losses, has no early stopping or checkpoint
selection, and reports accuracy values (`96.7`, `98.9`, `98.7`) inconsistent with
the shown accuracy function, which returns a fraction in `[0, 1]`.

An audit-only, deterministic sample of 625 images from the archived augmented
dataset produced 93.12% top-1 accuracy and 92.15% macro F1. It is explicitly
not an independent test set: the original split cannot be reconstructed, and
the sample may overlap training. These numbers must not be presented as test
accuracy.

The Flask inference path does not match the training preprocessing exactly:
training uses `Resize(255) -> CenterCrop(224)`, while the app directly resizes
to `224 x 224`. The app also does not convert images to RGB, does not calculate
confidence, and fails on RGBA inputs. Four of five demo images were rejected by
the audit reproduction for this channel mismatch.

Evidence-based verdict: **D. INVALID / CANNOT BE VERIFIED** as a reliable
generalization claim. The model artifact itself is loadable and usable for
controlled inference, but the complete model-and-application pipeline is not
verified as reliable.

## 2. Project structure

The complete file inventory is in `_audit/inventory.csv`. It includes the 73
top-level project files plus the extracted dataset trees, for which every image
file was included in the inventory.

| Area | Verified contents | Status |
|---|---|---|
| `Flask Deployed App/app.py` | Flask routes, file upload, model invocation, CSV lookup | IMPLEMENTED, runtime import currently blocked by dependency mismatch |
| `Flask Deployed App/CNN.py` | CNN definition and hardcoded 39-index mapping | IMPLEMENTED |
| `Flask Deployed App/templates/` | Flask HTML templates | IMPLEMENTED by route references |
| `Flask Deployed App/disease_info.csv` | 39 disease display records | IMPLEMENTED; index order verified |
| `Flask Deployed App/supplement_info.csv` | 39 supplement records | IMPLEMENTED; used by result/market routes |
| `Model/Plant Disease Detection Code.ipynb` | Dataset construction, split, model, historical evaluation | PARTIALLY IMPLEMENTED; contains stale/errors and no reproducibility seed |
| `Model/plant_disease_model_1_latest.pt` | Saved state dictionary | IMPLEMENTED; strict load passed |
| `test_images/` | 43 image files used by notebook examples and audit | IMPLEMENTED as test/demo assets, not a labeled benchmark |
| `demo_images/` | 5 image files | IMPLEMENTED as demo assets; 4 are RGBA and fail the app tensor assumption |
| dataset archives | augmented and non-augmented PlantVillage-style variants | DATA ARTIFACTS; augmented variant matches notebook size |
| `README.md`, folder READMEs | Empty or no operational instructions | UNKNOWN/INSUFFICIENT |
| PDF and `Model/model.JPG` | Notebook/model documentation | REFERENCE; PDF is image-based and text extraction returned no usable text |
| `Flask Deployed App/requirements.txt` | Historical pinned dependency set | IMPLEMENTED for intended deployment, not compatible with current environment without the historical stack |

## 3. Actual model architecture

### Artifact and load result

| Property | Verified value |
|---|---|
| File | `Model/plant_disease_model_1_latest.pt` |
| Format | PyTorch `OrderedDict` state dictionary |
| State entries | 60 |
| Load | PASS; strict load reported all keys matched |
| File size | 210,409,423 bytes |
| Framework | PyTorch |
| Input tensor | `1 x 3 x 224 x 224` |
| Output tensor | `1 x 39` |
| Output type | Raw finite logits |
| Final activation | None; final layer is linear |
| Parameters | 52,595,399 total/trainable |
| Loss recorded in notebook | `nn.CrossEntropyLoss()` |
| Optimizer recorded in notebook | `torch.optim.Adam(model.parameters())` |

### Layer structure

1. Conv2d 3 -> 32, ReLU, BatchNorm2d(32), Conv2d 32 -> 32, ReLU,
   BatchNorm2d(32), MaxPool2d(2).
2. Conv2d 32 -> 64, ReLU, BatchNorm2d(64), Conv2d 64 -> 64, ReLU,
   BatchNorm2d(64), MaxPool2d(2).
3. Conv2d 64 -> 128, ReLU, BatchNorm2d(128), Conv2d 128 -> 128, ReLU,
   BatchNorm2d(128), MaxPool2d(2).
4. Conv2d 128 -> 256, ReLU, BatchNorm2d(256), Conv2d 256 -> 256, ReLU,
   BatchNorm2d(256), MaxPool2d(2).
5. Dropout(0.4), Linear 50,176 -> 1,024, ReLU, Dropout(0.4), Linear
   1,024 -> 39.

The checkpoint contains weights only. No optimizer state, epoch, seed,
training history, class-map artifact, or model-version metadata is embedded in
the saved checkpoint.

The audit environment has `torch 2.10.0+cpu`. Importing the installed
`torchvision` fails before the Flask app can start with
`RuntimeError: operator torchvision::nms does not exist`. The repository pins
historical `torch==1.8.1+cpu` and `torchvision==0.9.1+cpu`; those versions were
not installed or changed during this audit.

## 4. Dataset

The notebook output says `Dataset ImageFolder` with 61,486 images. This exactly
matches the `Plant_leaf_diseases_dataset_with_augmentation.zip` archive, so
that is the dataset variant used by the recorded notebook session. The
non-augmented archive contains 55,448 images.

### Counts and class imbalance

| Dataset variant | Images | Classes | Minimum class | Maximum class | Max/min ratio |
|---|---:|---:|---:|---:|---:|
| Without augmentation | 55,448 | 39 | 152 (`Potato___healthy`) | 5,507 (`Orange___Haunglongbing_(Citrus_greening)`) | 36.23x |
| With augmentation | 61,486 | 39 | 1,000 across several classes | 5,507 (`Orange___Haunglongbing_(Citrus_greening)`) | 5.51x |

The augmented archive has 6,038 more files and appears to contain offline
augmentation/balancing. The notebook itself has no runtime augmentation
transform: it only uses `Resize`, `CenterCrop`, and `ToTensor`.

### Decode and format audit

Both archives were decoded fully with PIL and had zero unreadable files.

| Archive | Valid | Invalid | Dimensions | Modes | Formats |
|---|---:|---:|---|---|---|
| Without augmentation | 55,448 | 0 | 54,305 at 256x256; 1,143 at 256x192 | 55,447 RGB; 1 RGBA | 55,447 JPEG; 1 PNG |
| With augmentation | 61,486 | 0 | 55,409 at 256x256; 4,564 at 204x204; 1,143 at 256x192; 370 at 350x350 | 61,485 RGB; 1 RGBA | 61,485 JPEG; 1 PNG |

### Duplicate audit

Exact SHA-256 comparisons found:

- non-augmented archive: 22 exact duplicate groups, 23 duplicate extra files;
- augmented archive: 190 exact duplicate groups, 212 duplicate extra files;
- cross-variant comparison: 55,425 exact duplicate groups across the two
  extracted variants, covering 110,928 files.

The cross-variant result means the augmented and non-augmented archives cannot
be treated as independent datasets. The 190 duplicate groups inside the
augmented archive are directly relevant to any random split of the notebook's
61,486-image dataset. No perceptual near-duplicate analysis was performed.

## 5. Training pipeline

Reconstructed from notebook cells and saved outputs:

| Stage | Verified value |
|---|---|
| Dataset loader | `torchvision.datasets.ImageFolder("Dataset", transform=transform)` |
| Dataset used in notebook output | 61,486 images, 39 classes |
| Resize | `transforms.Resize(255)`; shortest side to 255 while preserving aspect ratio |
| Crop | `transforms.CenterCrop(224)` |
| Tensor conversion | `transforms.ToTensor()`; scales uint8 pixels to `[0, 1]` |
| Color | ImageFolder/PIL loader normally supplies RGB; no explicit custom color conversion in notebook |
| Runtime augmentation | None in transform; any augmentation is already embedded in the augmented archive |
| Index split | `floor(0.85 * 61486) = 52263` |
| Train | 36,584 images |
| Validation | 15,679 images |
| Actual test sampler | 9,223 images (`61486 - 52263`) |
| Notebook printed test size | 24,902; this is a calculation error (`61486 - 36584`) |
| Randomization | `np.random.shuffle(indices)` |
| Seed | Not set in notebook; split is not reproducible |
| Batch size | 64 |
| Loss | CrossEntropyLoss; raw logits expected |
| Optimizer | Adam defaults; explicit learning rate not recorded, so PyTorch default 0.001 is implied by the code version, not separately documented |
| Training call | 5 epochs |
| Validation during training | `validation_loader` is evaluated inside `batch_gd`; the `test_loader` argument is ignored by that function |
| Early stopping | Not implemented |
| Scheduler | Not implemented |
| Checkpointing | Save cell is commented out; later load uses an externally present `plant_disease_model_1_latest.pt` |
| Best epoch | Not available |

The dataset path in the notebook is `Dataset`, but the repository's extracted
directories have different names. Reproducing the notebook requires a manual
path arrangement that is not documented in the project.

## 6. Evaluation protocol

The notebook defines train, validation, and test samplers, but does not save
their memberships, seed, or split manifest. Therefore the original test set
cannot be reconstructed from the repository and the recorded `Test Accuracy`
cannot be independently verified.

An audit-only sample was evaluated from the archived augmented dataset:

- sorted archive members within each class;
- every 100th image per class;
- 625 images total, with every one of the 39 classes represented;
- canonical audit preprocessing was `Resize(255) -> CenterCrop(224) -> RGB ->
  ToTensor` to reproduce the training transform;
- model loaded from the unchanged checkpoint;
- top-1 and top-3 logits converted to softmax only for audit reporting.

This is labeled **AUDIT TEST SET** in `_audit/metrics.json`. It is not
independent because it was selected from the same archived dataset family and
the original random split is unavailable. It must not be used as a claim of
generalization.

## 7. Verified metrics

### Notebook-recorded values

The notebook output literally records:

```text
Train Accuracy : 96.7
Test Accuracy : 98.9
Validation Accuracy : 98.7
```

The shown `accuracy()` function returns `n_correct / n_total`, not a percentage.
The recorded values are therefore internally inconsistent with the visible
function and are not accepted as verified metrics. No actual test loss,
precision, recall, F1, top-3, or epoch-level history is present.

### Audit-only measured sample

| Metric | Audit-only result | Independent? |
|---|---:|---|
| Accuracy / top-1 | 93.12% | No |
| Macro precision | 93.11% | No |
| Macro recall | 92.22% | No |
| Macro F1 | 92.15% | No |
| Weighted precision | 93.77% | No |
| Weighted recall | 93.12% | No |
| Weighted F1 | 93.07% | No |
| Top-3 accuracy | 98.88% | No |
| Errors | 43 / 625 | No |

The complete classification report is `_audit/classification_report.csv`; the
confusion matrix is `_audit/confusion_matrix.csv`; raw metrics and protocol are
in `_audit/metrics.json`.

## 8. Confusion matrix analysis

The audit sample produced 43 top-1 errors. The most frequent observed pairs,
each occurring twice, were:

- Potato Early blight -> Tomato Septoria leaf spot;
- Tomato Late blight -> Tomato Yellow Leaf Curl Virus;
- Grape Black rot -> Grape Esca;
- Apple healthy -> Apple scab;
- Tomato Early blight -> Tomato Late blight;
- Tomato Late blight -> Tomato Septoria leaf spot;
- Apple healthy -> Blueberry healthy.

No confusion pair has been inferred from disease names; all pairs above come
from measured predictions. The complete matrix is in the CSV artifact.

Error structure in the audit sample:

| Error type | Count |
|---|---:|
| Same-crop errors | 23 |
| Cross-crop errors | 20 |
| Disease -> disease | 32 |
| Disease -> healthy | 4 |
| Healthy -> disease | 5 |
| Healthy -> healthy | 2 |

These dimensions overlap and describe the same 43 errors from different views.

## 9. Per-class performance

Lowest recall in the audit-only sample:

| Class | Recall | F1 | Support |
|---|---:|---:|---:|
| Tomato Early blight | 50.00% | 62.50% | 10 |
| Tomato Late blight | 65.00% | 68.42% | 20 |
| Potato Early blight | 70.00% | 82.35% | 10 |
| Apple healthy | 70.59% | 82.76% | 17 |
| Tomato Septoria leaf spot | 77.78% | 71.79% | 18 |
| Potato Late blight | 80.00% | 88.89% | 10 |
| Pepper bell Bacterial spot | 80.00% | 88.89% | 10 |

Lowest F1 was Tomato Early blight (62.50%), followed by Tomato Late blight
(68.42%), Tomato Septoria leaf spot (71.79%), and Apple scab (75.00%). Several
classes achieved 100% on this small sample, but that is not evidence of perfect
class performance; supports ranged from 10 to 56 and this is not an independent
test.

## 10. High-confidence error analysis

For the audit sample, softmax probabilities were computed only after model
inference for analysis. The application itself does not compute or display
confidence.

| Incorrect confidence threshold | Error count |
|---:|---:|
| >= 0.80 | 13 |
| >= 0.85 | 11 |
| >= 0.90 | 6 |
| >= 0.95 | 4 |

The full error records include path, actual class, predicted class, confidence,
and top-3 predictions in `_audit/error_analysis.csv`. The >=0.80 subset is in
`_audit/high_confidence_errors.csv`.

Examples of observed high-confidence errors include Tomato Yellow Leaf Curl
Virus -> Raspberry healthy at approximately 0.99997 in the filename-labeled
real-image set, and multiple archive errors above 0.95 in the audit sample.
The exact list and values, not this summary, are authoritative.

## 11. Confidence analysis

The application calculates no confidence: it applies `np.argmax` directly to
raw logits. The audit harness applied softmax only to quantify the model's
score behavior. No calibration method, temperature scaling, reliability
diagram, expected calibration error, or threshold policy exists in the project.

Audit sample confidence buckets:

| Bucket | Count | Accuracy |
|---|---:|---:|
| <0.50 | 16 | 18.75% |
| 0.50-0.60 | 20 | 65.00% |
| 0.60-0.70 | 14 | 64.29% |
| 0.70-0.80 | 22 | 77.27% |
| 0.80-0.90 | 29 | 75.86% |
| 0.90-0.95 | 22 | 90.91% |
| 0.95-1.00 | 502 | 99.20% |

Mean score was 0.9631 for correct predictions and 0.6602 for incorrect ones.
The sample suggests useful ranking signal but does not establish calibration.
High score is not certainty.

## 12. Inference pipeline verification

### Intended Flask flow

`multipart upload -> request.files['image'] -> original filename ->
static/uploads/<filename> -> PIL open -> resize(224,224) -> TF.to_tensor ->
reshape(1,3,224,224) -> CNN logits -> argmax -> disease_info.csv row ->
supplement_info.csv row -> HTML result`

### Verified implementation details

- The Flask application loads `CNN.CNN(39)` and the `.pt` state dictionary at
  import time.
- It uses `Image.open(image_path)` and `image.resize((224, 224))`.
- It uses `torchvision.transforms.functional.to_tensor` and then forcibly
  reshapes to `(1, 3, 224, 224)`.
- It selects `np.argmax(output)`.
- It does not apply softmax and returns no confidence.
- It maps the index to the CSV rows by positional indexing.
- The 39-row CSV order was verified against the hardcoded 39-class mapping;
  class index mapping is PASS for the available evidence.

## 13. Real image testing

The audit ran all 48 files under `test_images/` and `demo_images/` through the
model using the app-style direct-resize preprocessing, with a channel check
matching the app's 3-channel reshape.

- 44 images produced model predictions.
- 4 demo PNGs failed before inference because they are RGBA and the app assumes
  exactly 3 channels.
- Of 43 test images whose expected class could be inferred from the filename,
  35 matched and 8 mismatched. This is **not** a benchmark: the repository's
  README files are empty, and filenames are not independently verified labels.
- 5 demo images had no determinable expected class.

Observed filename-derived mismatches:

| Image | Prediction | Filename-derived expected class |
|---|---|---|
| `test_images/Apple_scab.JPG` | Tomato Septoria leaf spot | Apple scab |
| `test_images/apple_black_rot.JPG` | Pepper bell healthy | Apple black rot |
| `test_images/pepper_bacterial_spot.JPG` | Pepper bell healthy | Pepper bell bacterial spot |
| `test_images/tomato-bacterial-spot2.jpg` | Squash powdery mildew | Tomato bacterial spot |
| `test_images/tomato-leaf-curl-virus3.jpg` | Raspberry healthy | Tomato yellow leaf curl virus |
| `test_images/tomato-mold.jpg` | Potato early blight | Tomato leaf mold |
| `test_images/tomato_bacterial_spot.JPG` | Tomato early blight | Tomato bacterial spot |
| `test_images/tomato_yellow_leaf_curl_virus2.jpg` | Strawberry leaf scorch | Tomato yellow leaf curl virus |

Complete per-image predictions and top-3 values are in
`_audit/real_image_predictions.csv` and `.json`.

## 14. Data leakage findings

**FOUND / RISK CONFIRMED.**

The augmented archive used by the notebook contains 190 exact duplicate groups
internally. The two archived variants share 55,425 exact content hashes, so
they cannot be treated as independent evaluation sources. The notebook then
randomly shuffles the augmented dataset without a seed and does not remove
duplicates or group related files before splitting.

This does not prove that every recorded validation/test error was leaked, and it
does not prove exact duplicate placement across the lost split. It does prove
that a clean, reproducible, leakage-controlled evaluation was not established.

## 15. Overfitting findings

**POSSIBLE, not conclusively measurable.**

The notebook records a train output of 96.7 and a validation output of 98.7,
but those values are internally inconsistent with the visible accuracy
function's fraction return and are therefore not accepted as measured
percentages. No per-epoch loss/accuracy history is available, the checkpoint's
training epoch is unknown, and there is no early stopping record.

The audit sample's 93.12% result is lower than the notebook's claimed 98.7
validation value, but it is not independent and was sampled from the same
archive family. This is a warning, not a valid generalization-gap estimate.

## 16. Preprocessing verification

**MISMATCH FOUND.**

| Stage | Training | Flask app | Result |
|---|---|---|---|
| Resize | Preserve aspect ratio to short side 255 | Directly force 224 x 224 | MISMATCH |
| Crop | Center crop to 224 x 224 | No crop | MISMATCH |
| Color | ImageFolder/PIL RGB behavior | No explicit RGB conversion | PARTIAL/MISMATCH |
| Scale | ToTensor divides by 255 | TF.to_tensor divides by 255 | MATCH for valid RGB |
| Layout | NCHW tensor | CHW then reshape to NCHW | MATCH only for 3-channel input |
| Augmentation | None at runtime; pre-augmented archive | None | MATCH at runtime |

The RGBA demo failure is direct evidence of the color/channel mismatch. The
application's `view((-1, 3, 224, 224))` cannot safely accept grayscale or RGBA
images.

## 17. Main problems

| Priority | Issue | Evidence | Impact | Exact location | Recommended next action |
|---:|---|---|---|---|---|
| P0 | No verifiable independent test | No split manifest/seed; notebook output only | Generalization claim cannot be established | Notebook cells 9-17, outputs cell 51 | Build a frozen provenance-controlled test set before model changes |
| P0 | Production preprocessing mismatch | Training crop pipeline differs from app direct resize | Predictions can change between validation and app use | Notebook cell 6; `app.py` lines 19-23 | Align inference to the verified training transform |
| P0 | Current Flask environment cannot import app | `torchvision::nms` missing with installed torch 2.10/torchvision | Application cannot start in this environment | `app.py` line 4; requirements pins historical versions | Recreate the intended dependency environment; do not silently upgrade during this audit |
| P1 | No confidence output/calibration | App uses argmax on raw logits | Users cannot distinguish uncertainty or high-confidence errors | `app.py` lines 23-25 | Define and validate a confidence/calibration policy |
| P1 | Unreproducible split | `np.random.shuffle` with no seed; split lost | Metrics cannot be reproduced | Notebook cell 14 | Persist seed and split manifest |
| P1 | Duplicate contamination risk | 190 internal exact duplicate groups in augmented archive | Random split may overestimate performance | Augmented archive; `_audit/decode_with_augmentation.json` | Group/remove duplicates before a new evaluation |
| P1 | Reported metrics inconsistent | Accuracy function returns fraction; output says 96.7/98.9/98.7 | README/claims cannot be trusted | Notebook cells 49-51 | Recompute from an auditable fixed split |
| P1 | Test loss function ignores test loader | `test_laoder` parameter is unused; validation loader is evaluated | Training history mislabeled as test behavior | Notebook cell 37 | Separate train/validation/test evaluation logic |
| P2 | RGB/channel handling incomplete | RGBA demo images fail; grayscale would not satisfy reshape | Valid user images can error | `app.py` lines 19-22 | Normalize input color mode before tensor conversion |
| P2 | File upload path unsafe/brittle | Original filename used in `static/uploads`; relative CSV/model paths | Overwrites/path issues and cwd-dependent startup | `app.py` lines 11-15, 50-53 | Review upload validation and path handling after audit approval |
| P2 | Checkpoint provenance incomplete | State dict has no epoch/seed/optimizer metadata | Cannot establish which training run produced it | `.pt` artifact | Record reproducible model metadata with future artifacts |

## 18. Exact files/locations responsible

- Architecture and class mapping: `Flask Deployed App/CNN.py`, class `CNN`,
  final linear layer around line 52, `idx_to_classes` below it.
- Model loading: `Flask Deployed App/app.py:14-15`.
- App preprocessing and argmax: `Flask Deployed App/app.py:18-25`.
- Upload save path: `Flask Deployed App/app.py:50-53`.
- Training transform: notebook cell 6.
- Dataset loading: notebook cell 7.
- Split arithmetic: notebook cells 10-17.
- Model definition: notebook cell 29.
- Loss and optimizer: notebook cell 35.
- Training loop and validation misuse: notebook cell 37.
- DataLoader creation: notebook cell 39.
- Five-epoch training call: notebook cell 40.
- Commented save and external model load: notebook cells 42 and 44.
- Accuracy function and recorded outputs: notebook cells 49-51.
- Single-image inference: notebook cells 57 onward.

## 19. What should be improved later

This audit does not implement these changes. The evidence-supported next work
would be:

1. Recreate the intended historical environment or a tested compatible
   environment without silently changing the model.
2. Establish a fixed, deduplicated, provenance-controlled independent test set.
3. Recompute all metrics from a persisted split manifest, including per-class
   metrics, top-3, confusion matrix, and confidence calibration.
4. Make app preprocessing identical to the training transform and explicitly
   handle RGB conversion.
5. Add model provenance metadata, confidence semantics, input validation, and
   safe upload paths.
6. Re-evaluate the unchanged checkpoint after the pipeline is aligned before
   considering retraining.

## 20. What should NOT be changed based on this audit alone

- Do not replace the CNN architecture merely because the audit found pipeline
  and evaluation weaknesses.
- Do not retrain against the same unverified split and call the result better.
- Do not use the 625-image audit-only score as a production or independent test
  score.
- Do not treat softmax scores as calibrated probabilities.
- Do not delete either dataset archive, the checkpoint, or historical notebook.
- Do not mix this legacy 39-class model with the separate PlantCare 38-class
  model.

## 21. Final evidence-based verdict

**D. INVALID / CANNOT BE VERIFIED.**

This status applies to the complete model-performance claim, not to the basic
checkpoint load. The artifact passes structural loading and controlled
inference, but the project does not establish a reproducible independent test,
the reported notebook metrics are internally inconsistent, exact duplicates are
present, and the deployed inference preprocessing differs from training.

## Audit artifacts

- `_audit/metrics.json`
- `_audit/classification_report.csv`
- `_audit/confusion_matrix.csv`
- `_audit/error_analysis.csv`
- `_audit/high_confidence_errors.csv`
- `_audit/real_image_predictions.csv`
- `_audit/real_image_predictions.json`
- `_audit/audit_summary.json`
- `_audit/inventory.csv`
- `_audit/archive_metadata.json`
- `_audit/decode_without_augmentation.json`
- `_audit/decode_with_augmentation.json`
- `_audit/cross_archive_duplicates.json`
- `_audit/model_load.json`
- `_audit/notebook_outputs.json`
- `_audit/pdf_text.txt` and `_audit/pdf_page_1.png`
