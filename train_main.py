import os
import tensorflow as tf
import sonnet as snt
from datetime import datetime
from dataloader import ACREDataset
from SpikingCNN import SpikingCNN
from SpikingRNN import CustomALIF, spike_function
from Controller import Controller
import losses as ls


scnn_checkpoint_path = "model_weights1/scnn/scnn.ckpt"
scnn_checkpoint_dir = os.path.dirname(scnn_checkpoint_path)

rnn_checkpoint_path = "model_weights1/rnn/rnn.ckpt"
rnn_checkpoint_dir = os.path.dirname(rnn_checkpoint_path)

controller_checkpoint_path = "model_weights1/controller/controller.ckpt"
controller_checkpoint_dir = os.path.dirname(controller_checkpoint_path)

data_path = 'images'

def train_model(n_epochs, cnn_learning_rate, rnn_learning_rate, rnn_units, logging=True):
	"""
	Trains a model using the ACRE dataset.

	Args:
		n_epochs (int): The number of training epochs.
		cnn_learning_rate (float): The learning rate for the CNN optimizer.
		rnn_learning_rate (float): The learning rate for the RNN optimizer.
		rnn_units (int): The number of units in the RNN layer.
		logging (bool, optional): Whether to enable logging. Defaults to True.
	"""
	
	# Set up TensorBoard writer
	if logging:
		current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
		train_log_dir = 'logs02/' + current_time + '/train'
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

	batch_size = 10
	n_time = 10
	n_actions = 3
	eps = 1

	scnn = SpikingCNN()
	scnn_state = scnn.zero_state(batch_size, tf.float32)
	cnn_optimizer = tf.keras.optimizers.Adam(cnn_learning_rate)
	
	core = CustomALIF(n_in=n_inputs, n_rec=rnn_units)
	core_state = core.zero_state(batch_size, tf.float32)
	rnn_optimizer = tf.keras.optimizers.Adam(rnn_learning_rate)

	controller = Controller(n_actions, rnn_units)

	cce = tf.keras.losses.CategoricalCrossentropy()

	n_context = 6
	n_query = 4

	# Create functions for the loops
	for n_e in range(n_epochs):
		dataloader = ACREDataset(batch_size, dataset='train', data_path= data_path)
		n_batches = dataloader.n_batches

		for n_x in range(n_batches):
			total_loss = 0
			total_loss_for_cnn = 0

			with tf.GradientTape(persistent=True) as tape:
				images, labels = dataloader.get_next_batch()

				images = snt.BatchApply(snt.Conv2D(16, 8, stride=4))(images)
				context_images, query_images = tf.split(images, [6, 4], 1)

				for t in range(n_context):
					scnn_output, scnn_state = scnn(context_images[:, t], scnn_state)
					core_output, core_state = core(scnn_output[2], core_state)

				context_state_cnn = scnn_state
				context_state_rnn = core_state

				for t in range(n_query):
					scnn_output, scnn_state = scnn(query_images[:, t], context_state_cnn)
					core_output, core_state = core(scnn_output[2], context_state_rnn)

					logits, action, baseline = controller(core_output[0])

					value_targets = cce(tf.one_hot(labels[:, t], depth=n_actions), tf.one_hot(action, depth=n_actions))
					advantages = value_targets - baseline

					loss, loss_for_cnn = ls.calc_losses(logits, action, advantages, value_targets, baseline, scnn_output, core_output)

					total_loss += (loss * (1. / batch_size))
					total_loss_for_cnn += loss_for_cnn

					baseline_loss = ls.compute_baseline_loss(advantages)
					entropy_loss = ls.compute_entropy_loss(logits)
					pg_loss = ls.compute_policy_gradient_loss(logits, action, advantages)
					reg_loss = ls.compute_reg_loss(scnn_output, core_output)

				rnn_params = core.variable_list
				cnn_params = [*scnn.trainable_variables, *controller.trainable_variables]

			rnn_grads = tape.gradient(total_loss, rnn_params)
			cnn_grads = tape.gradient(total_loss_for_cnn, cnn_params)
			rnn_optimizer.apply_gradients(zip(rnn_grads, rnn_params))
			cnn_optimizer.apply_gradients(zip(cnn_grads, cnn_params))

		if logging:
			with train_summary_writer.as_default():
				tf.summary.scalar('total_loss', total_loss, step=n_e)
				tf.summary.scalar('total_loss_for_cnn', total_loss_for_cnn, step=n_e)
				tf.summary.scalar('baseline_loss', tf.reduce_sum(baseline_loss), step=n_e)
				tf.summary.scalar('entropy_loss', tf.reduce_sum(entropy_loss), step=n_e)
				tf.summary.scalar('pg_loss', tf.reduce_sum(pg_loss), step=n_e)
				tf.summary.scalar('reg_loss', reg_loss, step=n_e)

		scnn.save_weights(scnn_checkpoint_path)
		core.save_weights(rnn_checkpoint_path)
		controller.save_weights(controller_checkpoint_path)

		print("Epoch: {}, Loss: {}".format(n_e, total_loss))

	if logging:
		train_summary_writer.close()

	
train_model(n_epochs=1, cnn_learning_rate=1e-4, rnn_learning_rate=1e-4, rnn_units=1000, logging=True)	
	
	