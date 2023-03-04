# ProtoPNet: Are Learned Concepts Understandable?
A study on the interpretability of the concepts learned by [Prototypical Part Networks](https://arxiv.org/abs/1806.10574) (ProtoPNets). 

This work exploits the part locations annotations available for two different datasets to provide an objective evalution of the prototypes. An additional *diversity regularization* is also introduced to produce more diverse concepts. 

More details on the implementation can be found in the [report](report.pdf).

<table>
    <tr align="center">
        <td valign="top">
            <div><i>California Gull</i> class<div>
            <img src="img/cub200/alignment_matrix_prototypes.png" width="100%">
            <br/>
            <img src="img/cub200/1_prototype_580_bbox.jpg" width="18%"/>
            <img src="img/cub200/4_prototype_583_bbox.jpg" width="18%"/>
            <img src="img/cub200/6_prototype_585_bbox.jpg" width="18%"/>
            <img src="img/cub200/8_prototype_587_bbox.jpg" width="18%"/>
            <img src="img/cub200/9_prototype_588_bbox.jpg" width="18%"/>
            <br/>
                             <sup>580</sup><img width="6.5%"/>
            <img width="6.5%"/><sup>583</sup><img width="6.5%"/>
            <img width="6.5%"/><sup>585</sup><img width="6.5%"/>
            <img width="6.5%"/><sup>587</sup><img width="6.5%"/>
            <img width="6.5%"/><sup>588</sup>
        </td>
        <td valign="top">
            <div><i>Female</i> class<div>
            <img src="img/celeb_a/alignment_matrix_prototypes.png" width="100%">
            <br/>
            <img src="img/celeb_a/prototype_2_bbox.jpg" width="18%"/>
            <img src="img/celeb_a/prototype_3_bbox.jpg" width="18%"/>
            <img src="img/celeb_a/prototype_4_bbox.jpg" width="18%"/>
            <img src="img/celeb_a/prototype_6_bbox.jpg" width="18%"/>
            <img src="img/celeb_a/prototype_7_bbox.jpg" width="18%"/>
            <br/>
                             <sup>2</sup><img width="8%"/>
            <img width="8%"/><sup>3</sup><img width="8%"/>
            <img width="8%"/><sup>4</sup><img width="8%"/>
            <img width="8%"/><sup>6</sup><img width="8%"/>
            <img width="8%"/><sup>7</sup>
        </td>
    </tr>
</table>


## Get started
- Clone the repository and install the required dependencies:
    ```shell
    git clone https://github.com/materight/explainable-ProtoPNet.git
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

After training, the learned prototypes can be further pruned:
```shell
python prune_prototypes.py --dataset [data_path] --model [model_path]
```

## Evaluate learned prototypes
To evaluate a trained model and the learned prototypes, run:
```shell
python evaluate.py --model [model_path] {global|local|alignment} --dataset [data_path] 
```
- `global`: retrieve for each prototype the most activated patches in the whole dataset.
- `local`: evaluate the model on a subset of samples and generate visualizations for the activated prototypes for each class.
- `alignment`: generate plots for the alignment matrix of each class.

## Acknowledgments
This implementation is based on the original [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet) repository.
