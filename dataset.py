import datasets
import requests

class ImageDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description="A raw image dataset",
            features=datasets.Features({
                "image": datasets.Image()  # Images only, no labels
            }),
        )

    def _split_generators(self, dl_manager):
        repo_api_url = "https://api.github.com/repos/L3iCodes/dataset_loader/contents/images"
        response = requests.get(repo_api_url)

        if response.status_code == 200:
            image_urls = [file["download_url"] for file in response.json() if file["name"].endswith((".jpg", ".png"))]
        else:
            raise Exception("Error fetching image list from GitHub")

        # Download images locally
        downloaded_images = dl_manager.download(image_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"image_files": downloaded_images},  # Pass downloaded image paths
            ),
        ]

    def _generate_examples(self, image_files):
        for i, img_path in enumerate(image_files):
            yield i, {"image": img_path}  # Now, img_path is a local file path
