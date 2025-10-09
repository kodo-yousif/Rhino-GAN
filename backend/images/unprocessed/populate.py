import gdown
import os

# II2S data examples
folder_url = "https://drive.google.com/drive/folders/15jsR9yy_pfDHiS9aE3HcYDgwtBbAneId"

# Output directory
output_dir = os.path.dirname(os.path.abspath(__file__))

# Download all files in the folder
gdown.download_folder(folder_url, output=output_dir, quiet=True)

print("✅ All II2S images samples downloaded")
