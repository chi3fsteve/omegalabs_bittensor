import datasets

# Load the existing video IDs from the dataset
existing_ids = set(datasets.load_dataset('omegalabsinc/omega-multimodal')['train']['youtube_id'])

# Save the existing video IDs to a flat file
EXISTING_IDS_FILE = "existing_ids.txt"
with open(EXISTING_IDS_FILE, "w") as f:
    f.write("\n".join(existing_ids))