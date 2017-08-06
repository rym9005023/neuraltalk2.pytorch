# Neuraltalk2-pytorch

There's something difference compared to neuraltalk2.
- Instead of using random split, we use [karpathy's split](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).
- Put resnet in the loop, instead of preprocessing.
- Use resnet152 provided by Pytorch; the same way as in knowing when to look (resize and crop).

# Requirements
Python 2.7 (no [coco-caption](https://github.com/tylin/coco-caption) version for python 3), pytorch 0.2

# Pretrained FC model.
Download pretrained model from [link](https://drive.google.com/drive/folders/0B7fNdx_jAqhtOVBabHRCQzJ1Skk?usp=sharing). You also need pretrained resnet which can be downloaded from [pytorch-resnet](https://github.com/ruotianluo/pytorch-resnet.git).

Then you can follow [this section](#markdown-header-caption-images-after-training).

# Train your own network on COCO
**(Almost identical to neuraltalk2)**

Great, first we need to some preprocessing. Head over to the `coco/` folder and run the IPython notebook to download the dataset and do some very simple preprocessing. The notebook will combine the train/val data together and create a very simple and small json file that contains a large list of image paths, and raw captions for each image, of the form:

```
[{ "file_path": "path/img.jpg", "captions": ["a caption", "a second caption of i"tgit ...] }, ...]
```

Once we have this, we're ready to invoke the `prepro_split.py` script, which will read all of this in and create a dataset (several hdf5 files and a json file). For example, for MS COCO we can run the prepro file as follows:

```bash
$ python scripts/prepro_labels.py --input_json .../dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
$ python scripts/prepro_images.py --input_json .../dataset_coco.json --output_h5 data/cocotalk --images_root ...
```

You need to download [dataset_coco.json](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Karpathy's homepage.

This is telling the script to read in all the data (the images and the captions), allocate the images to different splits according to the split json file, extract the resnet101 features (both fc feature and last conv feature) of each image, and map all words that occur <= 5 times to a special `UNK` token. The resulting `json` and `h5` files are about 30GB and contain everything we want to know about the dataset.

**Warning**: the prepro script will fail with the default MSCOCO data because one of their images is corrupted. See [this issue](https://github.com/karpathy/neuraltalk2/issues/4) for the fix, it involves manually replacing one image in the dataset.

**(Copy end.)**

```bash
$ python train_pytorch.py --input_json data/cocotalk.json  --input_label_h5 data/cocotalk_label.h5 --input_image_h5 data/cocotalk_image.h5 --beam_size 3 --learning_rate 4e-4  --save_checkpoint_every 6000 --val_images_use 5000 --finetune_cnn_after 20
```

The train script will take over, and start dumping checkpoints into the folder specified by `checkpoint_path` (default = current folder). For more options, see `opts.py`.



If you have tensorflow, you can run train.py instead of `train_tb.py`. `train_tb.py` saves learning curves by summary writer, and can be visualized using tensorboard.

The current command use scheduled sampling, you can also set scheduled_sampling_start to -1 to turn off scheduled sampling.

If you'd like to evaluate BLEU/METEOR/CIDEr scores during training in addition to validation cross entropy loss, use `--language_eval 1` option, but don't forget to download the [coco-caption code](https://github.com/tylin/coco-caption) into `coco-caption` directory.

**A few notes on training.** To give you an idea, with the default settings one epoch of MS COCO images is about 7500 iterations. 1 epoch of training (with no finetuning - notice this is the default) takes about 15 minutes and results in validation loss ~2.7 and CIDEr score of ~0.5. ~~By iteration 50,000 CIDEr climbs up to about 0.65 (validation loss at about 2.4).~~

### Caption images after training

## Evaluate on test split of coco dataset

```bash
$ python eval.py --dump_images 0 --num_images 5000 --model model.pth --cnn_model_path model-cnn.pth --language_eval 1 --infos_path infos_<id>.pkl
```

The defualt split to evaluate is test. The default inference method is greedy decoding (`--sample_max 1`), to sample from the posterior, set `--sample_max 0`.

**Beam Search**. Beam search can increase the performance of the search for argmax decoding sequence. However, this is a little more expensive. To turn on the beam search, use `--beam_size N`, N should be greater than 1.

