# RAG Pipeline Package
import os

# 100% local: switch off library telemetry before any of these libraries load.
# onnxruntime >= 1.29 (used by fastembed) otherwise sends usage events to
# mobile.events.data.microsoft.com on every model load. Set a variable to 0
# in the environment or .env to opt back in.
for _var in ("ORT_DISABLE_TELEMETRY", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
    os.environ.setdefault(_var, "1")
