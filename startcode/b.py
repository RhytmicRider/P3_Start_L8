from transformers import pipeline
from PIL import Image
import requests

# Afbeelding laden
url = "https://codefeverpublic.blob.core.windows.net/public-content/images/0b9b1b2d16384249b8a7a67d32dc48aa.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Model laden
detector = pipeline("object-detection", model="facebook/detr-resnet-50", device_map="auto")

# Objecten detecteren en printen
resultaten = detector(image)
print(resultaten)


gefilterde_resultaten = []
for resultaat in resultaten:
  if resultaat["score"] > 0.7:
    gefilterde_resultaten.append(resultaat)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Plot afbeelding
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(image)

# Rechthoeken tekenen
for res in gefilterde_resultaten:
    label = res["label"]
    score = res["score"]
    box = res["box"]
    xmin = box["xmin"]
    ymin = box["ymin"]
    xmax = box["xmax"]
    ymax = box["ymax"]

    x = xmin
    y = ymin
    hoogte = ymax - ymin
    breedte = xmax - xmin
    tekst = f"{label}: {score:2f}"

    # Rechthoek tekenen
    rect = patches.Rectangle((x, y), breedte, hoogte, edgecolor="red", facecolor="none")
    ax.add_patch(rect)

    # Label + score tekenen
    ax.text(x, y, tekst)

plt.axis("off")
plt.show()