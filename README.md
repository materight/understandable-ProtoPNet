# Explainable ProtoPNet

## Get started
- Clone the repository and install the required dependencies:
    ```shell
    git clone https://github.com/materight/https://github.com/materight/explainable-ProtoPNet.git
    cd explainable-ProtoPNet
    pip install -r requirements.txt
    ```
- Download and prepare the data, either for the [Caltech-UCSD Birds-200](http://www.vision.caltech.edu/datasets/cub_200_2011/) or the [CelebAMask HQ](http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html) datasets:
    ```shell
    python prepare_data.py cub200
    python prepare_data.py celeb_a
    ```

## Train a model
TODO

## Evaluate learned prototypes
TODO

## Acknowledgments
This code is based on the original [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet) implementation.
