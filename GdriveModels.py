import os
import gdown
import zipfile

def download_google_drive_folder():

    """
    Download a folder from Google Drive and extract it to a local directory.

    Args:
        folder_id (str): The ID of the Google Drive folder to download.
        destination_folder (str): Local path to save the extracted folder content.
    """
    folder_id = "10A0JYCMhQ_4X75_eUHnF2cbm7NWPNUYl"
    destination_folder = "models_gdrive"  
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Google Drive URL for downloading folders
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing"
    print(f"Downloading folder from: {folder_url}")

    # Convert folder download link to export URL
    zip_url = f"https://drive.google.com/uc?id={folder_id}&export=download"

    # Local zip file path
    zip_file_path = os.path.join(destination_folder, "models_folder.zip")

    # Download the zip file
    gdown.download(zip_url, zip_file_path, quiet=False)

    # Extract the folder content
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

    # Remove the zip file
    os.remove(zip_file_path)
    print(f"Folder downloaded and extracted to: {destination_folder}")

