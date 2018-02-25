from clarifai.rest import ClarifaiApp
import matplotlib.image as mpimg
import numpy as np
from clarifai.rest import Image as ClImage
import time
import argparse
import StringIO

def dict_reader(concepts_list, preds_array):
    if args.target_model == 'moderation':
        preds_array[0]=filter(lambda concept: concept['name'] == 'safe', concepts_list)[0]['value']
        preds_array[1]=filter(lambda concept: concept['name'] == 'suggestive', concepts_list)[0]['value']
        preds_array[2]=filter(lambda concept: concept['name'] == 'explicit', concepts_list)[0]['value']
        preds_array[3]=filter(lambda concept: concept['name'] == 'drug', concepts_list)[0]['value']
        preds_array[4]=filter(lambda concept: concept['name'] == 'gore', concepts_list)[0]['value']
    elif args.target_model == 'nsfw-v1.0':
        preds_array[0]=filter(lambda concept: concept['name'] == 'sfw', concepts_list)[0]['value']
        preds_array[1]=filter(lambda concept: concept['name'] == 'nsfw', concepts_list)[0]['value']
    return preds_array

def nsfw_dict_reader(concepts_list, preds_array):
    preds_array[0]=filter(lambda concept: concept['name'] == 'sfw', concepts_list)[0]['value']
    preds_array[1]=filter(lambda concept: concept['name'] == 'nsfw', concepts_list)[0]['value']

def CW_est_batch(pred_plus_batch, pred_minus_batch, curr_target, max_index):
    logit_plus = np.log(pred_plus_batch)
    logit_plus_t = logit_plus[:, curr_target]
    logit_plus_max = logit_plus[:, max_index]
    logit_minus = np.log(pred_minus_batch)
    logit_minus_t = logit_minus[:, curr_target]
    logit_minus_max = logit_minus[:, max_index]

    logit_t_grad_est = (logit_plus_t - logit_minus_t) / delta / 2.0
    logit_max_grad_est = (logit_plus_max - logit_minus_max) / delta / 2.0
    return logit_max_grad_est - logit_t_grad_est

def xent_est_batch(pred_plus_batch, pred_minus_batch, curr_target):
    pred_plus_t = pred_plus_batch[:, curr_target]
    pred_minus_t = pred_minus_batch[:, curr_target]

    return (pred_plus_t - pred_minus_t) / delta / 2.0

def finite_diff_method(curr_sample, curr_target, p_t, max_index, U=None):
    # Randomly assign groups of group_size
    random_indices = np.random.permutation(dim)
    num_groups = dim / group_size
    print ('Num_groups: {}'.format(num_groups))
    group_indices = np.array_split(random_indices, num_groups)

    buffers = []

    for j in range(num_groups):
        # Create a perturbation for this group
        basis_vec = np.zeros((IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
        basis_vec_flat = basis_vec.reshape(-1)
        basis_vec_flat[group_indices[j]] = 1.

        # Generate perturbed images
        image_plus_i = np.clip(curr_sample + delta * basis_vec, CLIP_MIN, CLIP_MAX)
        image_minus_i = np.clip(curr_sample - delta * basis_vec, CLIP_MIN, CLIP_MAX)

        # Serialize perturbed images for submission
        buf_plus = StringIO.StringIO()
        mpimg.imsave(buf_plus, np.round(image_plus_i).astype(np.uint8), format='png')
        buffers.append(buf_plus)
        buf_minus = StringIO.StringIO()
        mpimg.imsave(buf_minus, np.round(image_minus_i).astype(np.uint8), format='png')
        buffers.append(buf_minus)

    # Submit the perturbed images
    num_queries = num_groups * 2
    inputs = [ClImage(file_obj=buf) for buf in buffers]
    batch_size = 30
    num_batches = int(num_queries/batch_size)
    result = []
    if num_batches>0:
        for i in range(num_batches):
            curr_input = inputs[i*batch_size:(i+1)*batch_size]
            result.extend(model.predict(curr_input)['outputs'])
        curr_input = inputs[num_batches*batch_size:]
        result.extend(model.predict(curr_input)['outputs'])
    else:
        result.extend(model.predict(inputs)['outputs'])

    for buf in buffers:
        buf.close()

    # Extract the output
    pred_plus_batch = np.zeros((num_groups, num_classes))
    for pred_plus, output in zip(pred_plus_batch, result[0:num_queries:2]):
        dict_reader(output['data']['concepts'], pred_plus)
    pred_minus_batch = np.zeros((num_groups, num_classes))
    for pred_minus, output in zip(pred_minus_batch, result[1:num_queries:2]):
        dict_reader(output['data']['concepts'], pred_minus)

    # Do the actual finite difference gradient estimate
    group_grad_est = CW_est_batch(pred_plus_batch, pred_minus_batch, curr_target, max_index)
    grad_est = np.zeros((IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
    grad_est_flat = grad_est.reshape(-1)
    for indices, single_grad_est in zip(group_indices, group_grad_est):
        grad_est_flat[indices] = single_grad_est
            
    # Getting gradient of the loss
#     loss_grad = -1.0 * grad_est/p_t
    loss_grad = grad_est

    return loss_grad

parser = argparse.ArgumentParser()
parser.add_argument("target_image_name", help="Image to misclassify")
parser.add_argument("--target_model", type=str, default='moderation', 
                    help="target model for attack")
parser.add_argument("--eps", type=int, default=16, 
                    help="perturbation magnitude to use")
parser.add_argument("--num_iter", type=int, default=5, 
                    help="number of iterations to run")
parser.add_argument("--group_size", type=int, default=10000,
                    help="Number of features to group together")
parser.add_argument("--delta", type=float, default=1.0,
                    help="local perturbation")

args = parser.parse_args()

app = ClarifaiApp()

model = app.models.get(args.target_model)

time1 = time.time()
success = 0
avg_l2_perturb = 0
curr_image=args.target_image_name+'.jpg'
curr_sample = np.array(mpimg.imread(curr_image),dtype=float)
array_shape = curr_sample.shape
if len(curr_sample.shape)>2:
    curr_sample=curr_sample[:,:,:3]
else:
    curr_sample = curr_sample.reshape((array_shape[0],array_shape[1],1))

BATCH_SIZE=1
IMAGE_ROWS=curr_sample.shape[0]
IMAGE_COLS=curr_sample.shape[1]
NUM_CHANNELS=curr_sample.shape[2]
dim=IMAGE_ROWS*IMAGE_COLS*NUM_CHANNELS
delta=args.delta
CLIP_MIN=0
CLIP_MAX=255
group_size=args.group_size
eps=args.eps
norm='linf'
alpha = float(args.eps/args.num_iter)
if args.target_model == 'moderation':
    num_classes = 5
elif args.target_model == 'nsfw-v1.0':
    num_classes = 2

curr_prediction = np.zeros((num_classes))
image_cl=ClImage(file_obj=open(curr_image,'rb'))
curr_predict_dict = model.predict([image_cl])['outputs'][0]['data']['concepts']
curr_prediction = dict_reader(curr_predict_dict, curr_prediction)
orig_index = np.argmax(curr_prediction)
print("Original prediction: {}".format(curr_prediction))

temp_sample = curr_sample
temp_image = curr_image
curr_target = 0

for i in range(args.num_iter):
    image_cl=ClImage(file_obj=open(temp_image,'rb'))

    temp_prediction = np.zeros((num_classes))
    temp_predict_dict = model.predict([image_cl])['outputs'][0]['data']['concepts']
    temp_prediction = dict_reader(temp_predict_dict, temp_prediction)
    temp_logits = np.log(temp_prediction)
    max_index = np.argmax(temp_prediction)
    loss_value = temp_logits[orig_index] - temp_logits[curr_target]
    print('Current loss value: {}'.format(loss_value))
    print('Current prediction: {}'.format(temp_prediction))

    p_t = temp_prediction[curr_target]

    loss_grad = finite_diff_method(temp_sample, curr_target, p_t, max_index)

    # Getting signed gradient of loss
    if norm == 'linf':
        normed_loss_grad = np.sign(loss_grad)
    elif norm == 'l2':
        grad_norm = np.linalg.norm(loss_grad.reshape(dim))
        normed_loss_grad = np.zeros_like(curr_sample)
        normed_loss_grad = loss_grad/grad_norm

    # eps_mod = eps - args.alpha
    image_adv = temp_sample - alpha * normed_loss_grad
    r = np.clip(image_adv-curr_sample, -eps, eps)
    temp_sample = np.clip(curr_sample + r, CLIP_MIN, CLIP_MAX)
    temp_image = args.target_image_name+'temp.png'
    mpimg.imsave(temp_image, np.round(temp_sample).astype(np.uint8))

x_adv = args.target_image_name+'_adv_'+str(args.eps)+'_'+str(args.num_iter)+'_'+str(args.delta)+'_'+str(args.group_size)+'.png'
mpimg.imsave(x_adv, np.round(temp_sample).astype(np.uint8))

# Getting the norm of the perturbation
perturb_norm = np.linalg.norm((image_adv-curr_sample).reshape(dim))
perturb_norm_batch = np.mean(perturb_norm)
avg_l2_perturb += perturb_norm_batch

image_adv_cl=ClImage(file_obj=open(x_adv,'rb'))

adv_prediction = np.zeros((num_classes))
adv_predict_dict = model.predict([image_adv_cl])['outputs'][0]['data']['concepts']
adv_prediction = dict_reader(adv_predict_dict, adv_prediction)
adv_logits = np.log(adv_prediction)
loss_value = adv_logits[orig_index] - adv_logits[curr_target]
success += np.sum(np.argmax(adv_prediction) == curr_target)

success = 100.0 * float(success)

print('Final loss: {}'.format(loss_value))
print('Final prediction: {}'.format(adv_prediction))
print('Success: {}'.format(success))

ofile=open(args.target_image_name+'.txt','a')

ofile.write('eps: {}, num_iter: {}, group_size: {}, delta: {}, model: {} ---- success: {} \n'.format(eps, args.num_iter, args.group_size, args.delta, args.target_model, success))
ofile.write("Original prediction: {} \n".format(curr_prediction))
ofile.write("Final prediction: {}\n".format(adv_prediction))
ofile.close()

# success = 100.0 - success

time2 = time.time()
print('Average l2 perturbation: {}'.format(avg_l2_perturb))
print('Total time: {}'.format(time2-time1))