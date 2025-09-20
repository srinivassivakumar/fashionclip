from datasets import load_dataset

dataset = load_dataset(
    "lvwerra/deepfashion-inshop",
    cache_dir="D:/Sri/mymiroo/S-AI/fashionclip/datasets"
)
print(dataset)
