---
layout: post
title:  "Fooling Clarifai's NSFW model"
author: Arjun Nitin Bhagoji
categories: posts
---

[Clarifai][clarifai] also hosts a [NSFW][nsfw-model] (Not Safe For Work) model, which returns a confidence score indicating whether or not the image contains nudity. This model only has 2 output classes, SFW and NSFW. For the attack on this model, we mainly experiment with a 900x601 pixel RGB image of a man and a woman in a state of undress. 
<figure>
  <img src="{{site.baseurl}}/assets/nsfw-002.jpg" alt="my alt text" align="middle"/>
  <figcaption> NSFW image of a man and a woman in a state of undress (benign image).</figcaption>
</figure>

The image is clearly NSFW. When the API returns the classification outcome for this image, it returns the following dictionary:
```
{u'outputs': [{u'created_at': u'2017-10-18T15:30:28.277015034Z',
   u'data': {u'concepts': [{u'app_id': u'main',
      u'id': u'ai_KxzHKtPl',
      u'name': u'nsfw',
      u'value': 0.8090479},
     {u'app_id': u'main',
      u'id': u'ai_RT20lm2Q',
      u'name': u'sfw',
      u'value': 0.19095212}]},
   u'id': u'eeab0c3397d94cb8a5f29073e3e044e7',
   u'input': {u'data': {u'image': {u'base64': u'true',
      u'url': u'https://s3.amazonaws.com/clarifai-api/img2/prod/small/c0b5b6be99074a458dd6f7062f2452b8/f6a54b1477c54072b4bf93b7a86bc3ab'}},
    u'id': u'ed911f66dc0b4ebb9efc22a21ea0f33b'},
   u'model': {u'app_id': u'main',
    u'created_at': u'2016-09-17T22:18:59.955626Z',
    u'display_name': u'NSFW',
    u'id': u'e9576d86d2004ed1a38ba0cf39ecb4b1',
    u'model_version': {u'created_at': u'2016-09-17T22:18:59.955626Z',
     u'id': u'a6b3a307361c4a00a465e962f721fc58',
     u'status': {u'code': 21100,
      u'description': u'Model trained successfully'}},
    u'name': u'nsfw-v1.0',
    u'output_info': {u'message': u'Show output_info with: GET /models/{model_id}/output_info',
     u'type': u'concept',
     u'type_ext': u'concept'}},
   u'status': {u'code': 10000, u'description': u'Ok'}}],
 u'status': {u'code': 10000, u'description': u'Ok'}}
```
The model is 81% confident that the image is NSFW, as given in the 'concepts' output. This is in line with our expectations, as there is nudity in the image, but it is not very explicit. 

The Gradient Estimation attack can be used on the NSFW model as well, since it also returns confidence scores. However, since the image has a large number of pixels, the Finite Differences method would need about 1,600,000 queries to construct an adversarial sample. Thus, to reduce the number of queries, the random grouping method for query reduction is used with a group size of 10,000. An iterative attack with 5 steps and a total perturbation magnitude of 20 (maximum is 255) was used to generate the following adversarial image:
<figure>
  <img src="{{site.baseurl}}/assets/nsfw-002_adv_20_5_10000.jpg" alt="my alt text"/>
  <figcaption>Adversarial image with perturbation magnitude of 20.</figcaption>
</figure>
This image is classified as SFW with a confidence of 0.55. Thus, with no access to the model, and a barely perceptible perturbation, the classification outcome has been changed. This highlights the magnitude of the problem when it comes to the deployment of these models in sensitive settings.


[nsfw-model]: https://clarifai.com/models/nsfw-image-recognition-model/e9576d86d2004ed1a38ba0cf39ecb4b1
[clarifai]: https://clarifai.com

