
Fine tune HF's modernBERT-base as a text encoder for Contrastive Language-Image Pretraining (CLIP)!<br>
Use natural language to search for images.<br>

# How to Get Started

To use a pretrained model to search through a directory of images, go to demo.py. For training, see train.py.<br>

# Model Details
**Text encoder:** modernBERT-base<br>
https://huggingface.co/answerdotai/ModernBERT-base<br>
**Vision encoder:** IdeficsV3 variant extracted from HF's smolVLM!<br>
https://huggingface.co/blog/smolvlm<br>

# Model Description

ModernBERT-base-CLIP is a multimodal model for Contrastive Language-Image Pretraining (CLIP), designed to align text and image representations in a shared embedding space. 
It leverages a fine-tuned ModernBERT-base text encoder and a frozen vision encoder (extracted from SmolVLM) to generate embeddings, which are projected into a 512-dimensional space using
linear layers. The model enables natural language-based image retrieval and zero-shot classification by optimizing a contrastive loss, which maximizes the similarity between matching text-image pairs while minimizing the similarity for non-matching pairs. 
Training was conducted on the Flickr30k dataset, with one-shot evaluation performed on COCO images (... or your own!) using the demo.ipynb script.

# Datasets

flickr30k: https://huggingface.co/datasets/nlphuji/flickr30 (training)<br>
Coco-captioning: https://cocodataset.org/#captions-2015 (demo)<br>

# Training Procedure

Vision embeddings are precomputed and stored as .npy files.

The model is trained using the InfoNCE contrastive loss, which encourages positive pairs, i.e. matching text and image embeddings, to be closer in the shared embedding space while pushing negative pairs apart.


# Hardware

Nvidia 3080 Ti
