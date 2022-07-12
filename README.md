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
To train a new model on a dataset, run:
```shell
python train.py --dataset [data_path] --exp_name [experiment_name]
```
Additional options can be specified (run the script with `--help` to see the available ones).

After training, the learned protoypes can be further pruned to remove duplicates:
```shell
python prune_prototypes.py --dataset [data_path] --model [model_path]
```

## Evaluate learned prototypes
TODO

## Acknowledgments
This implementation is based on the original [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet) repository.
