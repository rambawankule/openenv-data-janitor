# Swapping the RT-DETR Model in the VSS Alert Verification Profile

> **Applies to:** `MODE=2d_cv` (Alert Verification profile, `-m verification`)  
> **Working directory:** `deployments/developer-workflow/dev-profile-alerts/`

The "Alert Verification" profile (`bp_developer_alerts_2d_cv`) runs a **DeepStream perception pipeline** using RT-DETR as its primary detection model. The model is loaded inside the `perception-alerts` container. Swapping it means touching **five things** in order.

---

## Architecture Overview

```
RTSP stream
    │
    ▼
perception-alerts (DeepStream + RT-DETR ONNX)
    │  detects objects → publishes to Kafka (mdx-raw)
    ▼
vss-behavior-analytics-alerts
    │  applies business rules → emits alert candidates
    ▼
alert-bridge / vlm-as-verifier  ←── alert_type_config.json
    │  VLM confirms / rejects each candidate
    ▼
Elasticsearch  →  VSS Agent  →  UI
```

Your custom RT-DETR model replaces the ONNX at the **first stage** only. All downstream stages remain unchanged, except you may need to update the labels and (optionally) the VLM verifier prompts.

---

## Prerequisites

- Your custom RT-DETR model exported to **ONNX format**, matching the DeepStream TAO parser contract:
  - Output blobs: `pred_boxes` and `pred_logits`
  - Parser: `NvDsInferParseCustomDDETRTAO` (already compiled in the container via `libnvds_infercustomparser_tao.so`)
- The ONNX file placed somewhere accessible by your host, e.g.:
  ```
  $MDX_DATA_DIR/models/rtdetr-its/my_custom_model.onnx
  ```
- The **Alert Verification** profile is currently down (or you will restart it after changes).

---

## Step 1 — Place Your Model File

Copy your exported ONNX to the data directory the compose file already mounts:

```bash
# Default mount path in compose.yml (line 133):
# $MDX_DATA_DIR/models/rtdetr-its/model_epoch_035.fp16.onnx  →  container path

cp /your/export/dir/my_custom_model.onnx \
   $MDX_DATA_DIR/models/rtdetr-its/my_custom_model.onnx
```

> [!IMPORTANT]
> The container mounts the **entire file by name** (not the directory). You must either:
> - **Option A (simplest):** replace the existing filename → rename your file to `model_epoch_035.fp16.onnx`, or
> - **Option B (clean):** update the compose volume mount to point to your new filename (see Step 2b).

---

## Step 2 — Update the Compose Volume Mount (if using Option B)

**File:** `deployments/developer-workflow/dev-profile-alerts/compose.yml`  
**Line 133 (the ONNX volume mount):**

```yaml
# Before (default):
- $MDX_DATA_DIR/models/rtdetr-its/model_epoch_035.fp16.onnx:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/metropolis_perception_app/models/rtdetr-its/model_epoch_035.fp16.onnx

# After (your model):
- $MDX_DATA_DIR/models/rtdetr-its/my_custom_model.onnx:/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/metropolis_perception_app/models/rtdetr-its/my_custom_model.onnx
```

> [!NOTE]
> The GDINO model on line 134 is only used when `MODEL_NAME_2D=GDINO`. For the RT-DETR path, leave line 134 as-is (it is harmless when unused).

---

## Step 3 — Update the DeepStream Inference Config

**File:** `deployments/developer-workflow/dev-profile-alerts/deepstream/configs/rtdetr-960x544.txt`

This file is copied into the container image at build time (via `perception.Dockerfile`). Edit it **before** rebuilding.

```ini
[property]
gpu-id=0
offsets=0;0;0
net-scale-factor=0.00392156862745098

# ── CHANGE THESE TWO LINES ──────────────────────────────────────────────
labelfile-path=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/metropolis_perception_app/rtdetr-960x544-labels.txt
model-engine-file=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/metropolis_perception_app/models/rtdetr-its/my_custom_model.fp16.onnx_b30_gpu0_fp16.engine
onnx-file=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/metropolis_perception_app/models/rtdetr-its/my_custom_model.onnx
# ────────────────────────────────────────────────────────────────────────

batch-size=1
network-mode=2               # 2=FP16 (keep unless your model requires FP32)
network-type=0
num-detected-classes=<N>     # ← SET TO YOUR MODEL'S CLASS COUNT
interval=0
gie-unique-id=1
output-blob-names=pred_boxes;pred_logits
output-tensor-meta=1
infer-dims=3;<H>;<W>         # ← SET TO YOUR MODEL'S INPUT H x W (e.g. 3;544;960)
workspace-size=1048576
cluster-mode=4
strongly-typed=1
parse-bbox-func-name=NvDsInferParseCustomDDETRTAO
custom-lib-path=/opt/nvidia/deepstream/deepstream/lib/libnvds_infercustomparser_tao.so
maintain-aspect-ratio=1

[class-attrs-all]
pre-cluster-threshold=0.5    # ← adjust confidence threshold as needed
topk=20
```

**Key parameters to change:**

| Parameter | What to set |
|---|---|
| `onnx-file` | Container-side path to your new ONNX |
| `model-engine-file` | TRT engine cache path — use a new name so DeepStream rebuilds it |
| `num-detected-classes` | Exact number of classes your model outputs |
| `infer-dims` | `3;<height>;<width>` matching your model's input resolution |
| `labelfile-path` | Path inside container to your updated labels file |
| `pre-cluster-threshold` | Detection confidence threshold (0.0–1.0) |

> [!WARNING]
> If `model-engine-file` points to an existing `.engine` built for the old model, **DeepStream will use the cached engine and ignore your new ONNX**. Always change the engine filename when swapping models to force a rebuild.

---

## Step 4 — Update the Labels File

**File:** `deployments/developer-workflow/dev-profile-alerts/deepstream/configs/rtdetr-960x544-labels.txt`

The current file contains 5 classes (background + 4 ITS classes). Replace it with one class name per line matching your model's output indices:

```
# Example — replace with your actual classes (0-indexed, one per line)
background
class_0
class_1
class_2
...
```

> [!IMPORTANT]
> The number of lines must match `num-detected-classes` in `rtdetr-960x544.txt`. Class index 0 is `background` by convention for TAO RT-DETR models — confirm this matches your training setup.

---

## Step 5 — (Optional) Update the Run Config Input Dimensions

**File:** `deployments/developer-workflow/dev-profile-alerts/deepstream/configs/run_config-api-rtdetr-protobuf.txt`

If your model uses a **different input resolution** than 960×544, update the tracker and streammux dimensions to match:

```ini
[streammux]
# ...
width=<your_model_width>     # e.g. 1280
height=<your_model_height>   # e.g. 720

[tracker]
tracker-width=<your_model_width>
tracker-height=<your_model_height>
```

The `[primary-gie]` section at line 181 references `config-file=rtdetr-960x544.txt` — no change needed there unless you rename the inference config file.

---

## Step 6 — Update the VLM Verifier Prompts

**File:** `deployments/developer-workflow/dev-profile-alerts/vlm-as-verifier/configs/alert_type_config.json`

The VLM verifier fires after Behavior Analytics emits a candidate. The `alert_type` field must match the **category string** that Behavior Analytics emits when your new classes trigger a rule. Update the prompts to match your use case:

```json
{
  "version": "1.0",
  "alerts": [
    {
      "alert_type": "<BehaviorAnalytics_category_name>",
      "output_category": "<Display name in Elasticsearch/UI>",
      "prompts": {
        "system": "You are a helpful assistant.",
        "user": "<Yes/No question about the detected event. e.g. 'Is a forklift present near a pedestrian? Answer yes or no.'>"
      }
    }
  ]
}
```

> [!TIP]
> The `alert_type` must exactly match what Behavior Analytics writes to the `category` field of its Kafka message. Check your Behavior Analytics rules config under `dev-profile-alerts/vss-behavior-analytics/configs/` to confirm the category string.

---

## Step 7 — Rebuild and Restart

The `perception-alerts` container is **built from source** (not pulled), so config changes require a rebuild.

```bash
# From the repo root
cd deployments/developer-workflow/dev-profile-alerts

# 1. Tear down the running alerts profile
docker compose --env-file .env down perception-alerts perception-sdr-alerts

# 2. Rebuild the perception container (picks up new configs + labels)
docker compose --env-file .env build perception-alerts

# 3. Restart the full CV alerts profile
docker compose --env-file .env \
  --profile bp_developer_alerts_2d_cv \
  up -d
```

> [!NOTE]
> On first boot with a new ONNX, DeepStream will **build a new TensorRT engine** from the ONNX. This can take 10–30 minutes depending on model size and GPU. Watch the logs:
> ```bash
> docker logs -f perception-alerts
> ```
> Look for `Engine file built successfully` before expecting detections.

After changing `alert_type_config.json`, restart only the verifier:

```bash
docker restart alert-bridge   # or the container name for vlm-as-verifier
```

---

## Step 8 — Verify the Swap

```bash
# 1. Confirm the new ONNX is loaded (look for your filename in logs)
docker logs perception-alerts 2>&1 | grep -i "onnx\|engine\|model"

# 2. Check raw detections flowing through Kafka
docker exec -it mdx-kafka \
  kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic mdx-raw \
  --from-beginning | head -20

# 3. Check alert incidents via the VSS Agent
curl -s -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"input_message": "Show me recent alerts"}' | jq .
```

---

## Summary of Files Changed

| File | What you changed |
|---|---|
| `compose.yml` (line 133) | Volume mount path for ONNX file |
| `deepstream/configs/rtdetr-960x544.txt` | `onnx-file`, `model-engine-file`, `num-detected-classes`, `infer-dims` |
| `deepstream/configs/rtdetr-960x544-labels.txt` | Class names matching your model |
| `deepstream/configs/run_config-api-rtdetr-protobuf.txt` | `[streammux]` / `[tracker]` dims (only if resolution changed) |
| `vlm-as-verifier/configs/alert_type_config.json` | VLM verification prompts for new alert categories |

> [!CAUTION]
> Do **not** modify `config_triton_nvinferserver_gdino.txt` — that is for Grounding DINO (`MODEL_NAME_2D=GDINO`) and is not used in the RT-DETR path (`MODEL_NAME_2D` left unset or set to something other than `GDINO`).
