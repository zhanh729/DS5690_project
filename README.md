# DS5690_project

## Overview
### Motivation 
Have you ever been curious about your friends new shoes, or attracted by fancy shoes shows on ins photos? Time to try Nike shoes Recognizer! Traditional shoe classification systems struggle to differentiate among detailed and similar categories of Nike shoes. This project aims to improve classification precision by utilizing a Vision Transformer (ViT) model known for its effectiveness in handling complex image data.

In the past, the main approach to solve these tasks is Convolutional Neural Networks(CNNs). However, CNNs usually struggling when using on shoes recognization since shoes from the same brand usually have similar shape structure. And CNN perform weakly in extracting effective features for differentiation. Thus, it is usually not precise enough to classify shoes.

That's reason using the Vision Transformer (ViT) model, which is more suitable for handling long-distance dependencies and extracting global features through self-attention mechanisms.

### Model Introduction
The Vision Transformer (ViT) model was proposed in An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. 

Researchers show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

![alt text](vit.png)


The Vision Transformer (ViT) is a transformer encoder model (BERT-like) pretrained on a large collection of images in a supervised fashion, namely ImageNet-21k, at a resolution of 224x224 pixels. Next, the model was fine-tuned on ImageNet (also referred to as ILSVRC2012), a dataset comprising 1 million images and 1,000 classes, also at resolution 224x224.

Images are presented to the model as a sequence of fixed-size patches (resolution 16x16), which are linearly embedded. One also adds a [CLS] token to the beginning of a sequence to use it for classification tasks. One also adds absolute position embeddings before feeding the sequence to the layers of the Transformer encoder.

By pre-training the model, it learns an inner representation of images that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled images for instance, you can train a standard classifier by placing a linear layer on top of the pre-trained encoder. One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire image.
