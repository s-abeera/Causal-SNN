import os
from dataloader import ACREDataset
from datetime import datetime

from SpikingCNN import SpikingCNN
from SpikingRNN import CustomALIF, spike_function
from Controller import Controller

import tensorflow as tf 
import loss_functions as ls
import sonnet as snt

# Set up TensorBoard writer
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)


# Initialisation

n_kernel_1 = 8
n_kernel_2 = 8
n_stride_1 = 4
n_stride_2 = 4
n_filters_2 = 32

n_w_1 = (128 - n_kernel_1) // n_stride_1 + 2
n_w_2 = (n_w_1 - n_kernel_2) // n_stride_2 + 1
n_inputs = n_w_2 * n_w_2 * n_filters_2

batch_size = 5
n_time = 10
rnn_units = 100
learning_rate = 1e-6
n_actions = 3
# # 

scnn = SpikingCNN()
scnn_state = scnn.zero_state(batch_size, tf.float32)
cnn_optimizer = tf.keras.optimizers.Adam(learning_rate)

core = CustomALIF(n_in = n_inputs, n_rec = rnn_units)
core_state = core.zero_state(batch_size, tf.float32)
rnn_optimizer = tf.keras.optimizers.Adam(learning_rate)

controller = Controller(n_actions, rnn_units)

dataloader = ACREDataset(5, 'train')
n_batches = dataloader.n_batches

cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

n_context = 6
n_query   = 4

total_loss = 0
total_loss_for_cnn = 0

for n_x in range(n_batches):
	
	with tf.GradientTape(persistent = True) as tape:
		
		# Forward Pass
		images, labels = dataloader.get_next_batch()

		# Conv2d weights and SCNN weights are both 'cnn_variables'
		#conv_layer_1 = tf.keras.layers.Conv2D(16, 8, strides = (4, 4), padding= 'valid', trainable = True)
		#scnn.add(conv_layer_1)
		
		images = snt.BatchApply(snt.Conv2D(16, 8, stride=4))(images)
		#images = conv_layer_1(images)
		context_images, query_images = tf.split(images, [6, 4], 1)

		for t in range(n_context):
			
			scnn_output, scnn_state = scnn(context_images[:,t], scnn_state)
			core_output, core_state = core(scnn_output[2], core_state)
		
		context_state_cnn = scnn_state
		context_state_rnn = core_state
	
		for t in range(n_query):
			
			scnn_output, scnn_state = scnn(query_images[:,t], context_state_cnn)
			core_output, core_state = core(scnn_output[2], context_state_rnn)
			
			logits, action, baseline = controller(core_output[0])
			#value_targets = cce(labels[:,t], tf.cast(action, tf.float32))
			
			targets = labels[:, t]

			#advantages = value_targets - baseline
	
			loss = ls.calc_losses(logits, targets, scnn_output, core_output)

			total_loss += loss
			#total_loss_for_cnn += loss_for_cnn
			
			#Separate the losses
			#baseline_loss = ls.compute_baseline_loss(advantages)
			entropy_loss = ls.compute_entropy_loss(logits)
			#pg_loss = ls.compute_policy_gradient_loss(logits, action, advantages)
			reg_loss = ls.compute_reg_loss(scnn_output, core_output)
	
			# Log the losses to TensorBoard
			with train_summary_writer.as_default():
				tf.summary.scalar('total_loss', total_loss, step=n_x)
				tf.summary.scalar('total_loss_for_cnn', total_loss_for_cnn, step=n_x)
				#tf.summary.scalar('baseline_loss', tf.reduce_sum(baseline_loss), step=n_x)
				tf.summary.scalar('entropy_loss', tf.reduce_sum(entropy_loss), step=n_x)
				#tf.summary.scalar('pg_loss', tf.reduce_sum(pg_loss), step=n_x)
				tf.summary.scalar('reg_loss', reg_loss, step=n_x)

		rnn_params = core.variable_list		
		cnn_params = [*scnn.trainable_variables, *controller.trainable_variables]

	print('Batch:', n_x, 'Loss:', total_loss.numpy())
	# Calculate Gradients
	rnn_grads = tape.gradient(total_loss, rnn_params)
	#cnn_grads = tape.gradient(total_loss_for_cnn, cnn_params)

	clipped_gradients_rnn, _ = tf.clip_by_global_norm(rnn_grads, clip_norm=1.0)
	#clipped_gradients_cnn, _ = tf.clip_by_global_norm(cnn_grads, clip_norm=1.0)

	# Update Weights
	rnn_optimizer.apply_gradients(zip(clipped_gradients_rnn, rnn_params))
	#cnn_optimizer.apply_gradients(zip(clipped_gradients_cnn, cnn_params))

train_summary_writer.close()	
	

	
	
	
	
	
	
	
	