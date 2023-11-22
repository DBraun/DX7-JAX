import os
import requests
import argparse
import zipfile


def download_and_unzip(output_directory):

    url = "https://ccrma.stanford.edu/~braun/assets/DX7_AllTheWeb.zip"

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Download the zip file
    response = requests.get(url)
    if response.status_code == 200:
        zip_file_path = os.path.join(output_directory, "downloaded.zip")
        with open(zip_file_path, "wb") as file:
            file.write(response.content)

        # Unzip the file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(output_directory)

        # Clean up the zip file
        os.remove(zip_file_path)

        print("Download and unzip completed successfully.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


def main():
    parser = argparse.ArgumentParser(description="Download and unzip a zip file from the internet.")
    # parser.add_argument("url", help="URL of the zip file")
    parser.add_argument("-o", "--output-directory", default="dx7_patches", help="Output directory for downloaded and extracted files")
    args = parser.parse_args()

    download_and_unzip(args.output_directory)


if __name__ == "__main__":
    main()
