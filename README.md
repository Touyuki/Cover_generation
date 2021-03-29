This is a cover image generation system based on nerual networks. Here are some images we generated:
![image](https://github.com/Touyuki/Cover_generation/blob/main/images/good%20layout01.png)
![image](https://github.com/Touyuki/Cover_generation/blob/main/images/1.png)
![image](https://github.com/Touyuki/Cover_generation/blob/main/images/good%20layout02.png)
![image](https://github.com/Touyuki/Cover_generation/blob/main/images/2.png)
![image](https://github.com/Touyuki/Cover_generation/blob/main/images/good%20layout03.png)
![image](https://github.com/Touyuki/Cover_generation/blob/main/images/3.png)
![image](https://github.com/Touyuki/Cover_generation/blob/main/images/good%20layout04.png)
![image](https://github.com/Touyuki/Cover_generation/blob/main/images/4.png)
![image](https://github.com/Touyuki/Cover_generation/blob/main/images/good%20layout05.png)
![image](https://github.com/Touyuki/Cover_generation/blob/main/images/5.png)
![image](https://github.com/Touyuki/Cover_generation/blob/main/images/good%20layout06.png)
![image](https://github.com/Touyuki/Cover_generation/blob/main/images/6.png)
![image](https://github.com/Touyuki/Cover_generation/blob/main/images/good%20layout07.png)
![image](https://github.com/Touyuki/Cover_generation/blob/main/images/7.png)
![image](https://github.com/Touyuki/Cover_generation/blob/main/images/good%20layout08.png)
![image](https://github.com/Touyuki/Cover_generation/blob/main/images/8.png)

The structure is like:
![image](https://github.com/Touyuki/Cover_generation/blob/main/images/Structure.png)

![image](https://github.com/Touyuki/Cover_generation/blob/main/images/SRnet.png)


### Test

You can use this model by this command
```bash
python scripts/gui/simple-server.py --checkpoint models/YOUR_MODEL_CHECKPOINT 
```
We uploaded a pretrained weight file. You can download it and try the model. The verification code is ""

### Train

(Before the training you should download the coco images and annotation files, and put them into datasets/coco/images/ and  datasets/coco/images/annatations/)

1. Train the network
```bash
python train.py
```

2. Encode the appearance of the objects
```bash
python scripts/encode_features.py --checkpoint models/TRAINED_MODEL_CHECKPOINT
```

### References

We made use of the work of scene_generation(https://arxiv.org/abs/1909.05379 ICCV 2019). We modified their model mainly by adding an extra cover image discriminator and used another SRnet to generate text image with better quality. The pretrained SRnet is from https://github.com/Niwhskal/SRNet.

We used the COCO stuff 2017 dataset(https://cocodataset.org) for the training. In order to generate the solid regions (background regions with simple colors) of a book cover. We processed the coco images with real cover images. Both the cover images used in the processing and used in the cover image discriminator are from https://github.com/uchidalab/book-dataset
