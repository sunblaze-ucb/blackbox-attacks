---
layout: post
title: "Gallery of attack images: Clarifai's Moderation and NSFW Models"
author: Arjun Nitin Bhagoji
categories: posts
galleries:
 # gallery number one
 1:
   # row one in gallery one
   -
     - { url: '/assets/gore_3_25.png', alt: 'gore 1'}
     - { url: '/assets/gore_3_25_adv_16_5_1.0_2000.png', alt: 'gore 1 adv'}
   # row two in gallery one
   -
     - { url: '/assets/moderation-003_25.jpg', alt: 'gore 2'}
     - { url: '/assets/moderation-003_25_adv_16_5_1.0_5000.png', alt: 'gore 2 adv'}

 # gallery number two
 2: 
   # row one in gallery two
   -
     - { url: '/assets/moderation-004.jpg', alt: 'suggestive 1'}
     - { url: '/assets/moderation-004_adv_16_5_1.0_10000.png', alt: 'suggestive 1 adv'}
   # row two in gallery two
   -
     - { url: '/assets/suggestive_1.jpg', alt: 'suggestive 2'}
     - { url: '/assets/suggestive_1_adv_16_5_1.0_1000.png', alt: 'suggestive 2'}
 # gallery number three
 3: 
   # row one in gallery three
   -
     - { url: '/assets/nsfw-007.jpg', alt: 'nsfw 1'}
     - { url: '/assets/nsfw-007_adv_16_5_1.0_1000.png', alt: 'nsfw 1 adv'}
   # row two in gallery three
   -
     - { url: '/assets/nsfw-008.jpg', alt: 'nsfw 2'}
     - { url: '/assets/nsfw-008_adv_16_5_4.0_10000.png', alt: 'nsfw 2 adv'}
 4:
   -
     - { url: '/assets/moderation-001-small.jpg', alt: 'drug 1'}
     - { url: '/assets/moderation-001-small_adv_16_5_1.0_7670.png', alt: 'drug 1 adv'}
   -
     - { url: '/assets/Chunkyno3-small.jpg', alt: 'drug 2'}
     - { url: '/assets/Chunkyno3-small_adv_32_5_1.0_10047.png', alt: 'drug 2 adv'}
   -
     - { url: '/assets/CocaineUkelele-Opened-small.jpg', alt: 'drug 3'}
     - { url: '/assets/CocaineUkelele-Opened-small_adv_16_5_1.0_6997.png', alt: 'drug 3 adv'}
   -
     - { url: '/assets/Crack_Crack-small.jpg', alt: 'drug 3'}
     - { url: '/assets/Crack_Crack-small_adv_16_5_1.0_9464.png', alt: 'drug 3 adv'}
   -
     - { url: '/assets/Heroin-small.jpg', alt: 'drug 4'}
     - { url: '/assets/Heroin-small_adv_32_5_1.0_8746.png', alt: 'drug 4 adv'}

---

All of the original images are either sample images from [Clarifai][clarifai] or were found on the internet under various Creative Commons licenses which allow non-commercial reuse.

All adversarial images were generated using the Iterative Gradient Estimation attack with query reduction using random grouping. The number of iterations was set to 5 and the perturbation value to 16 for all images, except where otherwise indicated. The size of the random group was modified appropriately for each image to keep the number of queries low, while still allowing for the generation of adversarial images. In all image galleries, the original images appear first, immediately followed by their adversarial variants.


## Attacks on the Moderation model
The [Moderation model][content-moderation-api] hosted by Clarifai has 5 classes: 'safe', 'suggestive', 'explicit', 'drug' and 'gore'. 

### Images of gore

The images on the left were both originally classified as 'gore' by the Moderation model with a confidence of 1.0. The adversarial image corresponding to the first image is classified as 'safe' with a confidence of 0.69 while the second adversarial image is classified as 'safe' with a confidence of 0.55. The adversarial images need 110 and 200 queries to the model respectively and can be generated in under a minute.
{% include gallery.html  gallery=1 %}

### Suggestive images

The images below were originally classified as 'suggestive' by the Moderation model with a confidence of 0.72 (top, first image)and 0.99 (bottom, left image) respectively. The corresponding adversarial images (top, second image and bottom, right image) were classified as safe with a confidence of 0.58 and 0.79 respectively. The first adversarial image needed 810 queries while the second needed 1150 queries. These images took around 10 minutes each to generate.

{% include gallery.html  gallery=2 %}

### Images of drugs

The images on the left below were originally classified as 'drug' by the Moderation model with confidences (from top to bottom) of 1.000, 0.999, 0.987, 0.586, and 1.000.
The adversarial images on the right were classified (from top to bottom) as safe (with confidence 0.769), safe (0.549), explicit (0.525), and safe (0.759).
The adversarial images were generated with 197 queries each.

{% include gallery.html gallery=4 %}

## Attacks on the NSFW classification model
The [NSFW model][nsfw-api] hosted by Clarifai has just 2 classes: 'sfw' and 'nsfw'. The first image (topmost) is classified as 'nsfw' with a confidence of 0.83 by the NSFW model while the second image (third from top) is classified as 'nsfw' with a confidence of 0.85. The corresponding adversarial images are classifed as 'sfw' with confidences of 0.65 for the first image (second from top) and 0.59 for the second image (bottom). The adversarial images needed 1400 and 810 queries respectively. These images also took about 10 minutes to generate.

{% include gallery.html  gallery=3 %}


[content-moderation-api]: https://clarifai.com/models/moderation-image-recognition-model/d16f390eb32cad478c7ae150069bd2c6
[nsfw-api]: https://clarifai.com/models/nsfw-image-recognition-model/e9576d86d2004ed1a38ba0cf39ecb4b1
[clarifai]: https://clarifai.com