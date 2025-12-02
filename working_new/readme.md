📌 Deep Learning–Based Invisible Watermarking System
High-Quality Encoder–Decoder Model | PSNR 40+ | SSIM 0.98+ | TensorFlow | Google Colab

This project implements a deep-learning watermark embedding and extraction system using a lightweight Encoder–Decoder ResNet architecture. It embeds a watermark image into a host image invisibly and extracts it back with high accuracy.

The system is optimized to train on a single image, uses perceptual loss (VGG19) for high visual quality, and includes robustness augmentations such as noise and JPEG compression.

🚀 Features

✔ Invisible watermark embedding

✔ Accurate watermark extraction

✔ PSNR 40+ and SSIM 0.98+

✔ Uses VGG Perceptual Loss for high-quality images

✔ Lightweight Encoder–Decoder with residual blocks

✔ Supports saving models in H5, Keras, and SavedModel formats

✔ Works on Google Colab and local TensorFlow

✔ Can be converted to TFLite for mobile apps

📂 Project Structure
|-- encoder_model.h5
|-- decoder_model.h5
|-- encoder_model.keras
|-- decoder_model.keras
|-- encoder_saved_model/
|-- decoder_saved_model/
|-- watermarking.ipynb (optional)

📦 Installation

Run in Google Colab or locally:

pip install tensorflow pillow numpy matplotlib

📥 Upload Inputs

You will be asked to upload:

Host Image – the image into which the watermark is embedded

Watermark Image – the image you want to hide

Both are automatically resized to 256×256.

🔧 How It Works
1️⃣ Encoder

Takes image + watermark (6 channels)

Embeds watermark invisibly

Produces a watermarked image

2️⃣ Decoder

Takes only the watermarked image

Recovers the hidden watermark

🧠 Training

Training is fast (25 epochs):

Random augmentations

JPEG compression noise

Perceptual loss from VGG19

SSIM + MSE losses

This improves:

robustness

invisibility

extraction accuracy

📊 Evaluation Metrics

The following are displayed each epoch:

PSNR → Image quality

SSIM (Image) → Similarity to original

SSIM (Watermark) → Extraction quality

🖼 Output Images

After training, the script displays:

Original Image

Watermarked Image

Extracted Watermark

💾 Saving Models

Models are saved in three formats:

✔ H5 (legacy Keras)
encoder_model.h5
decoder_model.h5

✔ New Keras Format
encoder_model.keras
decoder_model.keras

✔ TensorFlow SavedModel (for deployment)
encoder_saved_model/
decoder_saved_model/


These can be used for:

Inference

Production apps

TFLite conversion

▶️ Using Saved Models (Example)

Embed watermark:

from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

encoder = load_model("encoder_model.h5", compile=False)

img = load_image("input.jpg")
wm  = load_image("watermark.png")

combined = np.expand_dims(np.concatenate([img, wm], axis=-1), 0)
watermarked = encoder.predict(combined)[0]

Image.fromarray((watermarked * 255).astype("uint8")).save("watermarked.jpg")


Extract watermark:

decoder = load_model("decoder_model.h5", compile=False)

watermarked = load_image("watermarked.jpg")
watermarked = np.expand_dims(watermarked, 0)

wm_out = decoder.predict(watermarked)[0]
Image.fromarray((wm_out * 255).astype("uint8")).save("extracted_wm.png")

📌 Future Improvements

Add TFLite conversion

Add mobile deployment (Android/iOS)

Add robustness attacks (blur, resize, crop)

Add CLI interface

🏁 Conclusion

This project provides a powerful and clean deep-learning watermarking system that:

Embeds watermarks invisibly

Extracts them reliably

Runs fast with excellent quality

Perfect for research, demos, and real-world applications.