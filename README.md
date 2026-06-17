# Object Identifier

A small web app that looks at a photo and tells you what's in it, using a pretrained image classification model.

## How it works

Upload a photo and the app runs it through MobileNetV2 (trained on ImageNet) to predict what's in the picture, showing the top 3 guesses with confidence scores.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal.

## Tech

- Python
- TensorFlow / Keras (MobileNetV2, pretrained on ImageNet)
- Streamlit for the interface

## Notes

The model recognizes around 1,000 everyday categories (animals, vehicles, household items, food, etc.) since that's what it was trained on. Clear, well-lit photos of a single object work best — busy scenes with several objects can confuse it.

If `pip install tensorflow` fails on your machine (common on Apple Silicon Macs), try `pip install tensorflow-macos` instead.
