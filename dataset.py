import datasets
import os

class ImageDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description="A raw image dataset",
            features=datasets.Features({
                "image": datasets.Image()  # Only images, no labels
            }),
        )

    def _split_generators(self, dl_manager):
        # Load image URLs directly from GitHub
        repo_url = "https://github.com/L3iCodes/dataset_loader/tree/main/images"
        image_files = [f"{repo_url}{img}" for img in ["image1.jpg", "image2.jpg", "image3.jpg"]]  # Modify dynamically

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,  # No actual split, just a single dataset
                gen_kwargs={"image_files": image_files},
            ),
        ]

    def _generate_examples(self, image_files):
        for i, img_path in enumerate(image_files):
            yield i, {"image": img_path}
