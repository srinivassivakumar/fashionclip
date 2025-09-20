import os
from datasets import load_dataset

# Download Fashion-MNIST dataset
print("Downloading Fashion-MNIST...")
ds = load_dataset("fashion_mnist")

# Where to save extracted images
save_dir = "D:/Sri/mymiroo/products/"
os.makedirs(save_dir, exist_ok=True)

print("Exporting first 200 images to", save_dir)

# Label mapping for Fashion-MNIST
label_names = {
    0: "T-shirt_top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle_boot"
}

# Use .select() to get proper dict samples
subset = ds["train"].select(range(200))

for i, example in enumerate(subset):
    img = example["image"]      # PIL.Image object
    label = label_names[example["label"]]
    filename = f"train_{i}_{label}.png"
    img.save(os.path.join(save_dir, filename))

print("✅ Done! 200 images saved in:", save_dir)
