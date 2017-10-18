from clarifai.rest import ClarifaiApp
import scipy.misc
import matplotlib.image as mpimg
import numpy as np
from clarifai.rest import Image as ClImage
import time
import argparse

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

def CW_est(x_plus_i, x_minus_i, curr_target, max_index):
    image_plus=ClImage(file_obj=open(x_plus_i,'rb'))
    pred_plus=np.zeros((num_classes))
    pred_plus_dict = model.predict([image_plus])['outputs'][0]['data']['concepts']
    pred_plus = dict_reader(pred_plus_dict, pred_plus)
    logit_plus = np.log(pred_plus)
    logit_plus_t = logit_plus[curr_target]
    logit_plus_max = logit_plus[max_index]

    image_minus=ClImage(file_obj=open(x_minus_i,'rb'))
    pred_minus=np.zeros((num_classes))
    pred_minus_dict = model.predict([image_minus])['outputs'][0]['data']['concepts']
    pred_minus = dict_reader(pred_minus_dict, pred_minus)
    logit_minus = np.log(pred_minus)
    logit_minus_t = logit_minus[curr_target]
    logit_minus_max = logit_minus[max_index]

    logit_t_grad_est = (logit_plus_t - logit_minus_t)/delta
    logit_max_grad_est = (logit_plus_max - logit_minus_max)/delta

    return logit_t_grad_est/2.0, logit_max_grad_est/2.0


def xent_est(x_plus_i, x_minus_i, curr_target):
    image_plus=ClImage(file_obj=open(x_plus_i,'rb'))
    pred_plus=np.zeros((num_classes))
    pred_plus_dict = model.predict([image_plus])['outputs'][0]['data']['concepts']
    pred_plus = dict_reader(pred_plus_dict, pred_plus)
    pred_plus_t = pred_plus[curr_target]
    
    image_minus=ClImage(file_obj=open(x_minus_i,'rb'))
    pred_minus=np.zeros((num_classes))
    pred_minus_dict = model.predict([image_minus])['outputs'][0]['data']['concepts']
    pred_minus = dict_reader(pred_minus_dict, pred_minus)
    pred_minus_t = pred_minus[curr_target]
    single_grad_est = (pred_plus_t - pred_minus_t)/delta
    print(single_grad_est)

    return single_grad_est/2.0

def finite_diff_method(curr_sample, curr_target, p_t, max_index, U=None):
    grad_est = np.zeros((IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
    random_indices = np.random.permutation(dim)
    num_groups = dim / group_size
    print ('Num_groups: {}'.format(num_groups))
    for j in range(num_groups):
        if j % 100 == 0:
            print j
        basis_vec = np.zeros((IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
        if j != num_groups-1:
            curr_indices = random_indices[j*group_size:(j+1)*group_size]
        elif j == num_groups-1:
            curr_indices = random_indices[j*group_size:]
        per_c_indices = curr_indices%(IMAGE_COLS*IMAGE_ROWS)
        channel = curr_indices/(IMAGE_COLS*IMAGE_ROWS)
        row = per_c_indices/IMAGE_COLS
        col = per_c_indices % IMAGE_COLS
        for i in range(len(curr_indices)):
            basis_vec[row[i], col[i], channel[i]] = 1.
        image_plus_i = np.clip(curr_sample + delta * basis_vec, CLIP_MIN, CLIP_MAX)
        x_plus_i = 'clarifai_images/moderation_image_plus.jpg'
        mpimg.imsave(x_plus_i,image_plus_i/255)
        image_minus_i = np.clip(curr_sample - delta * basis_vec, CLIP_MIN, CLIP_MAX)
        x_minus_i = 'clarifai_images/moderation_image_minus.jpg'
        mpimg.imsave(x_minus_i,image_minus_i/255)
        
#         single_grad_est = xent_est(x_plus_i, x_minus_i, curr_target)
        logit_t_grad_est, logit_max_grad_est = CW_est(x_plus_i, x_minus_i, curr_target, max_index)
        single_grad_est = logit_max_grad_est - logit_t_grad_est
        for i in range(len(curr_indices)):
            grad_est[row[i], col[i], channel[i]] = single_grad_est.reshape((1))
            
    # Getting gradient of the loss
#     loss_grad = -1.0 * grad_est/p_t
    loss_grad = grad_est

    return loss_grad

parser = argparse.ArgumentParser()
parser.add_argument("target_image_name", help="Image to misclassify")
parser.add_argument("--target_model", type=str, default='nsfw-v1.0', 
                    help="target model for attack")
parser.add_argument("--eps", type=int, default=16, 
                    help="perturbation magnitude to use")
parser.add_argument("--num_iter", type=int, default=5, 
                    help="number of iterations to run")
parser.add_argument("--group_size", type=int, default=10000,
                    help="Number of features to group together")
parser.add_argument("--delta", type=float, default=0.01,
                    help="local perturbation")

args = parser.parse_args()

app = ClarifaiApp(api_key='b9b1dc3d4d2f45f5a07978674eab670e')

model = app.models.get(args.target_model)

time1 = time.time()
success = 0
avg_l2_perturb = 0
curr_image=args.target_image_name+'.jpg'
curr_sample = np.array(mpimg.imread(curr_image),dtype=float)
curr_sample=curr_sample[:,:,:3]

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
    print('Max_index: {}'.format(max_index))
    loss_value = temp_logits[max_index] - temp_logits[curr_target]
    print('Current loss value: {}'.format(loss_value))

    p_t = temp_prediction[curr_target]

    loss_grad = finite_diff_method(temp_sample,curr_target, p_t, max_index)
    # np.save('loss_grad_drugs.npy',loss_grad)
    # loss_grad = np.load('loss_grad_drugs.npy')

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
    temp_image = args.target_image_name+'temp.jpg'
    mpimg.imsave(temp_image, temp_sample/255)

x_adv = args.target_image_name+'_adv_'+str(args.eps)+'_'+str(args.num_iter)+'_'+str(args.delta)+'_'+str(args.group_size)+'.jpg'
mpimg.imsave(x_adv, temp_sample/255)

# Getting the norm of the perturbation
perturb_norm = np.linalg.norm((image_adv-curr_sample).reshape(dim))
perturb_norm_batch = np.mean(perturb_norm)
avg_l2_perturb += perturb_norm_batch

image_adv_cl=ClImage(file_obj=open(x_adv,'rb'))

adv_prediction = np.zeros((num_classes))
adv_predict_dict = model.predict([image_adv_cl])['outputs'][0]['data']['concepts']
adv_prediction = dict_reader(adv_predict_dict, adv_prediction)
success += np.sum(np.argmax(adv_prediction) == curr_target)

success = 100.0 * float(success)

ofile=open(args.target_image_name+'.txt','a')

ofile.write('eps: {}, num_iter: {}, group_size: {}, delta: {}, model: {} ---- success: {} \n'.format(eps, args.num_iter, args.group_size, args.delta, args.target_model, success))
ofile.write("Original prediction: {} \n".format(curr_prediction))
ofile.write("Final prediction: {}\n".format(adv_prediction))
ofile.close()

# success = 100.0 - success

time2 = time.time()
print('Average l2 perturbation: {}'.format(avg_l2_perturb))
print('Total time: {}'.format(time2-time1))