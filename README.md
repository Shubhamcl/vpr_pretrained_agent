# Visual Place Recognition pre-trained end-to-end driving agent

  

Repository for the paper "Visual place recognition pre-training for end to end trained autonomous driving agent"


## Setup training code:
1. Clone this repository:
	```
	git clone https://github.com/Shubhamcl/vpr_pretrained_agent.git
	```

 2. Clone SegVPR repository:
	```
	git clone https://github.com/valeriopaolicelli/SegVPR.git
	```

2. Add the address of SegVPR repository's src folder address in model.py line 7:
	```
	seg_vpr_address  =  "/home/user/SegVPR/src/"
	```

## Data collection and Evaluation of trained model:  
We follow guidelines and settings from the [Imitating a Reinforcement Learning Coach](https://arxiv.org/abs/2108.08265) paper. To collect data and evaluate the same way, use the code available [here](https://github.com/zhejz/carla-roach/tree/main).

After data collection, run resizing of images on the collected data. Specify the address within the file `data_altering/resize_image.py` and run it.


## To train:

1. Configure configs/main_train.yaml file with settings of choice.
2. Run the following command:
	```
	python main.py
	```
## Citation
Please cite our work if you found it useful:
```
To be added
```
