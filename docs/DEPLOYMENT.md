# Deployment

## Export target: ONNX

`scripts/export_onnx.py` runs `model.export(format="onnx")` on the trained weights and then
**loads the result back and runs real inference on it**, comparing the ONNX predictions against
the PyTorch ones detection by detection. An export that is never reloaded proves nothing.

### Why ONNX and not something else

The deployment hardware for this project is the machine it was developed on: an Intel Core
i7-10510U with integrated graphics and **no CUDA device**. That rules most of the options out
before preference enters into it.

| Format | Verdict here |
|---|---|
| **ONNX** | **Chosen.** Runs on ONNX Runtime's CPU execution provider, which is exactly the available hardware. Portable — the same file runs on a Windows laptop, a Linux site box, or a Raspberry Pi, with no re-export. |
| TensorRT (`.engine`) | Requires an NVIDIA GPU. None present. |
| OpenVINO | A genuine contender on Intel silicon and often faster than ONNX Runtime there. Skipped to keep the dependency footprint small; the export call is a one-line change (`format="openvino"`) if a site deployment justifies it. |
| CoreML | Apple platforms only. |
| TFLite | Aimed at Android/embedded. Would be the right answer if this ran on a phone or a Coral device on a site pole — it does not. |
| Raw `.pt` | Works, but drags the full PyTorch install onto every deployment target. |

The honest summary: **ONNX is chosen because it matches the hardware that actually exists and
keeps the deployment portable.** On a real rollout to Intel edge boxes, benchmarking OpenVINO
against ONNX Runtime would be the first optimisation to try.

### Running the export

```bash
python scripts/export_onnx.py
```

Requires `models/best.pt` from the Colab training run. The script:

1. Loads the trained model and prints its task and class list.
2. Exports to ONNX at 640 px, reporting file size against the PyTorch original.
3. Copies the result to `models/best.onnx`, where the app looks for it.
4. Reloads it through `YOLO("models/best.onnx")` and predicts on a real site image.
5. Prints PyTorch and ONNX detections side by side, with the maximum confidence drift.

Small confidence drift between the two is expected — ONNX Runtime executes fp32 kernels in a
different order, so results agree to a few decimal places rather than bit-exactly. A change in
the *number* or *class* of detections would indicate a real export problem.

### Result

Run against the trained weights — full output in [`logs/10_onnx_export.log`](../logs/10_onnx_export.log):

| | |
|---|---|
| exported | `models/best.onnx`, opset 18, 10.57 MB (PyTorch: 5.47 MB) |
| runtime | ONNX Runtime 1.29.0, `CPUExecutionProvider` |
| detections | **7 on both**, identical class sequence |
| max confidence drift | **0.0120** |

The size roughly doubling is expected: the `.pt` file stores fp16 weights, while the ONNX graph is
fp32 with the ops written out explicitly.

Annotated ONNX output: `outputs/05_onnx_roundtrip.jpg`.

> The per-call timings the script prints are **not** a benchmark. The first ONNX call carries
> session initialisation and graph optimisation, so it reads far slower than PyTorch on a single
> image. Comparing steady-state throughput would need a warm-up pass and repeated timed runs,
> which is deliberately out of scope here — the claim being made is *correctness of the export*,
> not a speedup.

---

## The Streamlit app

```bash
streamlit run app.py
```

Two modes:

- **Image** — upload a site photo, get annotated detections, a sortable table (violations ranked
  first by severity) and a compliance verdict banner.
- **Video** — upload a clip, run tracking through the `ViolationLedger`, get a ranked incident
  table with durations plus the evidence crops.

The sidebar picks the model (custom PyTorch, custom ONNX, or COCO fallback) and sets the
confidence and NMS IoU thresholds live.

### Implementation notes worth knowing

- **`initial_sidebar_state="expanded"` is load-bearing.** Streamlit's default `"auto"` leaves the
  sidebar out of the DOM entirely on load, which hides the model picker and the confidence
  threshold — the two controls that change the verdict. This was caught by inspecting the
  rendered page, not by assuming an HTTP 200 meant the UI was correct.
- **`@st.cache_resource` on model loading.** Without it, every slider movement reloads the weights
  from disk and the app feels broken.
- **Graceful degradation.** `available_models()` offers whatever exists. With no trained weights
  the app still runs on COCO and says plainly that it cannot assess compliance, rather than
  silently reporting everyone as compliant — which is the dangerous failure mode for a safety tool.
- **Video is capped by a frame slider.** Tracking runs at roughly 4 fps on this CPU, so processing
  is bounded explicitly instead of appearing to hang.

### Verification

`logs/06_streamlit_boot.log` records the boot: server start, `HTTP 200` on `/` and on
`/_stcore/health`, the rendered sidebar and main-pane contents read out of the live DOM, and zero
`stException` elements on the page.

---

## What production would need beyond this

Stated so the scope of what is here is not overread. This is a working demonstrator, not a
deployed system:

- **Streaming input.** The app takes uploads. A real installation consumes RTSP from site cameras,
  which needs the threaded producer/consumer pattern from the course's video-engineering module —
  a synchronous `cap.read()` loop cannot keep up with a live stream.
- **Persistence.** Incidents live in a CSV for the duration of a run. A real system writes to a
  database with retention rules, because violation records are evidence.
- **Alerting.** `ultralytics.solutions.SecurityAlarm` covers email; a site would want the incident
  pushed to a supervisor's phone within seconds, not surfaced in a web page nobody has open.
- **Per-site calibration.** The confidence threshold selected in `03_evaluation.ipynb` is tuned on
  this dataset. Camera angle, lighting and helmet colours differ by site, so each installation
  needs its own threshold and ideally a few hundred labelled frames of its own footage.
- **Privacy.** Continuous worker monitoring carries obligations regardless of intent.
  `solutions.ObjectBlurrer` can blur faces while keeping the PPE assessment intact, and retention
  of the evidence crops needs a stated policy.
