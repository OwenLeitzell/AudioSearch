# Whisper MathBERT Pipeline Requirements

Script: `AIIR/fine-tuned-models/Pipelines/whisper_mathbert_pipeline.py`

## Python Packages
- `torch`
- `numpy`
- `pandas`
- `datasets`
- `transformers`
- `evaluate`
- `openpyxl`
- `tensorboard`

## Suggested Install
```bash
pip install torch numpy pandas datasets transformers evaluate openpyxl tensorboard
```

## External Model and Dataset Dependencies
- Hugging Face models:
  - `openai/whisper-base`
  - `tbs17/MathBERT`
- Hugging Face datasets:
  - `abby1492/mathbridge-audio`
  - `OwenLeitzell/FormulaSearch`

