<h1> EXITT_TARNet </h1>

<hr>

This repository contains the official source code for the "EXIT$^{2}$ : **EX**plainability **I**n **T**ime-series **T**ransformer" paper.


<h1> Requirements </h1>

- `pytorch`==1.12.1+cu113
- `captum`==0.7.0
- `tslearn`==0.6.3
- `numpy`==1.23.5
- `h5py`
- `scikit-learn`

<hr>

<h1> Training part </h1>

<h2> Qucick Start </h2>

This process utilizes the pre-trained model files(`check_points`) provided by the repository.

- command
    `python xai_test.py --model_path $your_model_path --dataset $dataset_name  --batch $batch_size  --nlayers $num_of_layers --emb_size $embedding_size --nhead $num_of_head --xai_method $xai_method --device $your_device`

- example
     `python xai_test.py --model_path ./check_points/BirdChicken_4_0.0001_2_64_8_0.900.mdl --dataset BirdChicken  --batch 4  --nlayers 2 --emb_size 64 --nhead 8 --xai_method random --device cuda:0`


<h2> Training from scratch </h2>

| A. 