# Setting up

1. I am using python 3.8.5

```
source env/bin/activate
pip install -r req.txt
```

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.23.08              Driver Version: 545.23.08    CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090        On  | 00000000:5E:00.0 Off |                  N/A |
|  0%   36C    P8              31W / 350W |     12MiB / 24576MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce RTX 3090        On  | 00000000:AF:00.0 Off |                  N/A |
|  0%   34C    P8              21W / 350W |     12MiB / 24576MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      2447      G   /usr/lib/xorg/Xorg                            4MiB |
|    1   N/A  N/A      2447      G   /usr/lib/xorg/Xorg                            4MiB |
+---------------------------------------------------------------------------------------+
```


2. Dataset Preparation
Output structure of the folders are shown below.
```
images/
├── metadata.json
└── train
    ├── conditional
    └── images
```


Sample metadata.json

> make sure the json has keys: image, conditioning_image, text - the names should be as it is.
> Also make sure that the path inside is global path as mentioned, not a root path.

```
{"status": "train", "image": "/media/cilab/data/NTIRE/flare/Flare7K/images/train/images/IMG_6275_284.jpg", "conditioning_image": "/media/cilab/data/NTIRE/flare/Flare7K/images/train/conditional/IMG_6275_284.jpg", "text": "Optimize the image for digital display by eliminating lens flares."}
{"status": "train", "image": "/media/cilab/data/NTIRE/flare/Flare7K/images/train/images/IMG_6216_239.jpg", "conditioning_image": "/media/cilab/data/NTIRE/flare/Flare7K/images/train/conditional/IMG_6216_239.jpg", "text": "Refine the image by reducing glare and improving visual clarity."}
{"status": "train", "image": "/media/cilab/data/NTIRE/flare/Flare7K/images/train/images/IMG_6236_63.jpg", "conditioning_image": "/media/cilab/data/NTIRE/flare/Flare7K/images/train/conditional/IMG_6236_63.jpg", "text": "Improve image quality by minimizing the presence of lens flares."}
```

3. editing the bash file for training
    Please edit the following things
    1. --output_dir, Outputpath
    2. --train_data_dir to be a json file not a directory due to changes in the code.
    3. --validation_image, --validation_prompt
    4. this will report to wandb


```
accelerate launch --mixed_precision="fp16" --multi_gpu train_controlnet.py   --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  --output_dir='./flaremodel_v1'  --train_data_dir=/media/cilab/data/NTIRE/flare/Flare7K/images/metadata.json  --resolution=512  --learning_rate=1e-5  --validation_image "/media/cilab/data/NTIRE/flare/Flare7K/images/train/conditional/IMG_20240205_223333_214.jpg" "/media/cilab/data/NTIRE/flare/Flare7K/images/train/conditional/20240205_223426_376.jpg"  --validation_prompt "Remove distracting glare from the image to enhance its overall appeal." "Restore the image to its intended state by removing any unintended lens flares."  --train_batch_size=1  --gradient_accumulation_steps=4  --tracker_project_name="controlnet-demo"  --report_to=wandb
```


4. train the model
```
bash train_control.sh
```


5. Inference the model
```
python inference.py
```