import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
import time

def xent_est(x_plus_i, x_minus_i, curr_target):
    image_plus=ClImage(file_obj=open(x_plus_i,'rb'))
    pred_plus=np.zeros((2))
    pred_plus[0]=model.predict([image_plus])['outputs'][0]['data']['concepts'][0]['value']
    pred_plus[1]=model.predict([image_plus])['outputs'][0]['data']['concepts'][1]['value']
    pred_plus_t = pred_plus[curr_target]

    image_minus=ClImage(file_obj=open(x_minus_i,'rb'))
    pred_minus=np.zeros((2))
    pred_minus[0]=model.predict([image_minus])['outputs'][0]['data']['concepts'][0]['value']
    pred_minus[1]=model.predict([image_minus])['outputs'][0]['data']['concepts'][1]['value']
    pred_minus_t = pred_minus[curr_target]
    single_grad_est = (pred_plus_t - pred_minus_t)/delta

    return single_grad_est/2.0

def finite_diff_method(curr_sample, curr_target, p_t):
    grad_est = np.zeros((IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS))
    random_indices = np.random.permutation(dim)
    num_groups = dim / group_size
    print num_groups
    for j in range(num_groups):
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
        x_plus_i = 'nsfw_image_plus.jpg'
        plt.imsave(x_plus_i,image_plus_i/255)
        image_minus_i = np.clip(curr_sample - delta * basis_vec, CLIP_MIN, CLIP_MAX)
        x_minus_i = 'nsfw_image_minus.jpg'
        plt.imsave(x_minus_i,image_minus_i/255)

        single_grad_est = xent_est(x_plus_i, x_minus_i, curr_target)
        print('{}, {}'.format(j, single_grad_est))
        for i in range(len(curr_indices)):
            grad_est[row[i], col[i], channel[i]] = single_grad_est.reshape((1))

    # Getting gradient of the loss
    loss_grad = -1.0 * grad_est/p_t

    return loss_grad

img=mpimg.imread('nsfw-002.jpg')

delta=0.01
CLIP_MIN=0
CLIP_MAX=255
group_size=1000
eps=8
norm='linf'

app = ClarifaiApp(api_key='f1c3261cde1f46e0892d6618d1af4881')

model = app.models.get('nsfw-v1.0')
time1 = time.time()
success = 0
avg_l2_perturb = 0
curr_image='nsfw-002.jpg'
curr_sample = np.array(mpimg.imread(curr_image),dtype=float)

IMAGE_ROWS=curr_sample.shape[0]
IMAGE_COLS=curr_sample.shape[1]
NUM_CHANNELS=curr_sample.shape[2]
dim=IMAGE_ROWS*IMAGE_COLS*NUM_CHANNELS

image_cl=ClImage(file_obj=open(curr_image,'rb'))

curr_prediction = np.zeros((2))
curr_prediction[0]=model.predict([image_cl])['outputs'][0]['data']['concepts'][0]['value']
curr_prediction[1]=model.predict([image_cl])['outputs'][0]['data']['concepts'][1]['value']

curr_target = np.argmax(curr_prediction)

p_t = curr_prediction[curr_target]

loss_grad = finite_diff_method(curr_sample,curr_target, p_t)
loss_grad_file = 'loss_grad_nsfw_2_'+str(delta)+'_'+str(group_size)+'_'+str(eps)+'.npy'
np.save(loss_grad_file,loss_grad)

# Getting signed gradient of loss
if norm == 'linf':
    normed_loss_grad = np.sign(loss_grad)
elif norm == 'l2':
    grad_norm = np.linalg.norm(loss_grad.reshape(dim))
    indices = np.where(grad_norm != 0.0)
    normed_loss_grad = np.zeros_like(curr_sample)
    normed_loss_grad[indices] = loss_grad[indices]/grad_norm[indices, None, None, None]

# eps_mod = eps - args.alpha
image_adv = np.clip(curr_sample + eps * normed_loss_grad, CLIP_MIN, CLIP_MAX)
x_adv = 'nsfw_adv_'+str(delta)+'_'+str(group_size)+'_'+str(eps)+'.jpg'
plt.imsave(x_adv,image_adv/255)

# Getting the norm of the perturbation
perturb_norm = np.linalg.norm((image_adv-curr_sample).reshape(dim))
perturb_norm_batch = np.mean(perturb_norm)
avg_l2_perturb += perturb_norm_batch

image_adv_cl=ClImage(file_obj=open(x_adv,'rb'))

adv_prediction = np.zeros((2))
adv_prediction[0]=model.predict([image_adv_cl])['outputs'][0]['data']['concepts'][0]['value']
adv_prediction[1]=model.predict([image_adv_cl])['outputs'][0]['data']['concepts'][1]['value']
success += np.sum(np.argmax(adv_prediction) == curr_target)

success = 100.0 * float(success)

success = 100.0 - success

time2 = time.time()
print('Success: {}'.format(success))
print('Original predict: {}'.format(curr_prediction))
print('Adversarial predict: {}'.format(adv_prediction))
print('Average l2 perturbation: {}'.format(avg_l2_perturb))
print('Total time: {}'.format(time2-time1))
