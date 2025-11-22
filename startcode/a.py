from transformers import pipeline

classificeerder = pipeline("image-classification", model="google/vit-base-patch16-224")

afbeeldingen = [
    "https://codefeverpublic.blob.core.windows.net/public-content/images/053153adb0d44173b7eb3713d0412666.jpg",
    "https://codefeverpublic.blob.core.windows.net/public-content/images/87a99ee4ad8e44189361ae74422eace4.png",
    "https://codefeverpublic.blob.core.windows.net/public-content/images/2e658c21a4e444ffb5447307ebc3abf2.png",
    "https://codefeverpublic.blob.core.windows.net/public-content/images/e6752850439b4226af4f89e8e0105523.jpg",
    "https://codefeverpublic.blob.core.windows.net/public-content/images/78e8ed9f70be49cc8b551b3317733555.png",
    "https://codefeverpublic.blob.core.windows.net/public-content/images/07c20246ab874864b04cab918e4e2ce9.jpg",
    "https://codefeverpublic.blob.core.windows.net/public-content/images/082985fdb14c435b9319d4bd89b24787.png",
    "https://codefeverpublic.blob.core.windows.net/public-content/images/3371f03ab00c46229238b3aae1c666b2.jpg",
    "https://codefeverpublic.blob.core.windows.net/public-content/images/fb0ce6e7b77e4a82a414cf304a92d3b7.jpg",
    "https://codefeverpublic.blob.core.windows.net/public-content/images/8a2f56caa7484f05ae9819ffa2d1e2a4.jpg"
]


for afbeeldingen in afbeeldingen:
    resultaat = classificeerder(afbeeldingen)
    label = resultaat[0]["label"]
    score = resultaat[0]["score"]
    print(label, str(score))