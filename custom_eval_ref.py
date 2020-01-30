"""
Infinite evaluation loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import argparse

#import cifar10_input
#from pgd_attack import LinfPGDAttack
from model import Model

import sys
sys.path.append('../foolbox')
import foolbox as fb

# Global constants
with open('config.json') as config_file:
  config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']
data_path = config['data_path']


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir')
args = vars(parser.parse_args())
patch_model_dir = str(args['model_dir'])

config['model_dir'] = patch_model_dir
model_dir = config['model_dir']

assert(patch_model_dir == config['model_dir'])
assert(patch_model_dir == model_dir)

print('continue with model_dir: {}'.format(model_dir))
#_ = input('>')

model = Model(mode='eval')

global_step = tf.contrib.framework.get_or_create_global_step()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float')
x_test = x_test.astype('float')

print('cifar10 x train shape: {}'.format(x_train.shape))
print('cifar10 y train shape: {}'.format(y_train.shape))
print('cifar10 x test shape: {}'.format(x_test.shape))
print('cifar10 x test shape: {}'.format(y_test.shape))

# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
  os.makedirs(eval_dir)

last_checkpoint_filename = ''
already_seen_state = False

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(eval_dir)

def report_acc_xent(x_data, y_data, model, sess, batches=1000):
    
    x_batches = np.array_split(x_data, batches)
    y_batches = np.array_split(y_data.flatten(), batches)

    t_acc = 0
    t_xent = 0
    for x_batch, y_batch in tqdm(zip(x_batches, y_batches)):

      dict_batch = {model.x_input: x_batch, model.y_input: y_batch}
      batch_acc, batch_xent = sess.run([model.num_correct, model.xent], feed_dict=dict_batch)

      t_acc += batch_acc
      t_xent += batch_xent

    print('acc: {}. xent: {}.'.format(t_acc/len(x_data), t_xent/len(x_data)))

def eval_robustness(x, y, model, sess):

  #epsilon_scan = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
  epsilon_scan = [8./255.]
  for epsilon in epsilon_scan:
    print('scanning epsilon: {}'.format(epsilon))
    #epsilon = 0.005
    #step_size = 0.1
    #iterations=5
    #random_start=False

    # cifar10 challenge ref
    # epsilon = 8./255.
    step_size = 2./255.
    iterations=20
    random_start=False

      #init foolbox model
    fb_model = fb.models.TensorFlowModel(model.x_input, model.pre_softmax, bounds=(0., 255.))

    attack = fb.attacks.LinfinityBasicIterativeAttack
    attack_distance = fb.distances.Linfinity

    x_adv = []

    #per adv class
    for y_class_idx in np.unique(y):
      print('x adv found so far: {}'.format(len(x_adv)))

      x_class = x[y == y_class_idx]
      y_class = y[y == y_class_idx]
      
      assert(len(x_class) == len(y_class))
      assert(len(x_class) == np.sum(y == y_class_idx)), '{} == {}'.format(len(x_class), np.sum(y == y_class))

      assert(len(x_class) == 1000)
      assert(len(y_class) == 1000)

      #loop over the samples in a batched manner
      x_batches = np.array_split(x_class, 100)
      y_batches = np.array_split(y_class, 100)

      #init the attack    
      #attack_criteria = fb.criteria.TopKMisclassification(3)

      #cifar10 challenge ref
      attack_criteria = fb.criteria.TopKMisclassification(1)
      fb_attack = attack(model=fb_model, criterion=attack_criteria, distance=attack_distance)

      for x_batch, y_batch in tqdm(zip(x_batches, y_batches)):  

        #get out mispreds and manually do everything else
        y_batch_pred = np.argmax(fb_model.forward(x_batch), axis=-1)

        incorrect_x = x_batch[y_batch_pred != y_batch]
        incorrect_y = y_batch[y_batch_pred != y_batch]

        correct_x = x_batch[y_batch_pred == y_batch]
        correct_y = y_batch[y_batch_pred == y_batch]

        assert(len(correct_x) == len(correct_y))
        assert(len(incorrect_x) == len(incorrect_y))

        x_class_adv = fb_attack(correct_x, correct_y, binary_search=False, epsilon=epsilon, stepsize=(epsilon/(0.3))*step_size, iterations=iterations, return_early=False, random_start=random_start)

        #cifar 10 challenge _ref
        #x_class_adv = fb_attack(correct_x, correct_y, binary_search=False, epsilon=epsilon, stepsize=step_size, iterations=iterations, return_early=False, random_start=random_start)

        x_class_adv = x_class_adv[~np.isnan(x_class_adv).any(axis=(1,2,3))]

        x_adv.extend(x_class_adv)
        x_adv.extend(incorrect_x)

    x_adv = np.array(x_adv)
    assert(~np.any(np.isnan(x_adv)))

    print('for e={}:{}'.format(epsilon, len(x_adv)))

# A function for evaluating a single checkpoint
def evaluate_checkpoint(filename):
  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, filename)

    #evaluate clean accuracy    

    #report_acc_xent(x_train, y_train, model, sess, batches=1000)
    report_acc_xent(x_test, y_test, model, sess, batches=200)

    eval_robustness(x_test.copy(), y_test.copy().flatten(), model, sess)

    exit()

# Infinite eval loop
dummy_cond = True
while dummy_cond:
  dummy_cond = False
  cur_checkpoint = tf.train.latest_checkpoint(model_dir)

  # Case 1: No checkpoint yet
  if cur_checkpoint is None:
    if not already_seen_state:
      print('No checkpoint yet, waiting ...', end='')
      already_seen_state = True
    else:
      print('.', end='')
    sys.stdout.flush()
    time.sleep(10)
  # Case 2: Previously unseen checkpoint
  elif cur_checkpoint != last_checkpoint_filename:
    print('\nCheckpoint {}, evaluating ...   ({})'.format(cur_checkpoint,
                                                          datetime.now()))
    sys.stdout.flush()
    last_checkpoint_filename = cur_checkpoint
    already_seen_state = False
    evaluate_checkpoint(cur_checkpoint)
  # Case 3: Previously evaluated checkpoint
  else:
    if not already_seen_state:
      print('Waiting for the next checkpoint ...   ({})   '.format(
            datetime.now()),
            end='')
      already_seen_state = True
    else:
      print('.', end='')
    sys.stdout.flush()
    time.sleep(10)
