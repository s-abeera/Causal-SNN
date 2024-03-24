import tensorflow as tf
import numpy as np
import numpy.random as rd


# # Define the graph for the loss function and the definition of the error
# def loss(labels, logits):
#     loss_pred = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
#     loss_pred = tf.reduce_mean(loss_pred)

#     loss_reg = reg_loss()
#     loss = loss_pred + loss_reg

#     prediction = tf.argmax(logits, axis=2)
#     is_correct = tf.equal(labels, prediction)
#     is_correct_float = tf.cast(is_correct, dtype=tf.float32)
#     ler = 1. - tf.reduce_mean(ler)
#     decoded = prediction

#     return loss, ler, decoded


def compute_entropy_loss(logits):
	
	policy = tf.nn.softmax(logits)
	log_policy = tf.nn.log_softmax(logits)
	
	entropy = -policy * log_policy
	negentropy = tf.reduce_sum(-entropy,1)

	return negentropy

def compute_reg_loss(scnn_output, core_output):
	
	# RNN Componenets
	thr = 1.0
	rate_cost = 50.
	voltage_cost_rnn = 0.0001
	voltage_cost_cnn = 0.5
	beta = 0.1
	thr_scnn = 0.1
	voltage_reg_method = 'avg_time'
	
	
	rnn_v = core_output[1][..., 0]
	rnn_thr = thr + beta * core_output[1][..., 1]
	rnn_pos = tf.nn.relu(rnn_v - rnn_thr)
	rnn_neg = tf.nn.relu(-rnn_v - rnn_thr)
	voltage_reg_rnn = tf.reduce_sum(tf.reduce_mean(tf.square(rnn_pos), 1))
	voltage_reg_rnn += tf.reduce_sum(tf.reduce_mean(tf.square(rnn_neg), 1))
	rnn_rate = tf.reduce_mean(core_output[0], (0, 1))
	rnn_mean_rate = tf.reduce_mean(rnn_rate)
	rate_loss = tf.reduce_sum(tf.square(rnn_rate - .02)) * 1.	

	# CNN Componenets
	conv1_z = scnn_output[0]
	conv2_z = scnn_output[2]
	lin_z = tf.zeros_like(scnn_output[1]) # likely wrong
	conv_1_rate = tf.reduce_mean(conv1_z, (0, 1))
	conv_2_rate = tf.reduce_mean(conv2_z, (0, 1))
	linear_rate = tf.reduce_mean(lin_z, (0, 1))
	mean_conv_1_rate = tf.reduce_mean(conv_1_rate)
	mean_conv_2_rate = tf.reduce_mean(conv_2_rate)
	mean_linear_rate = tf.reduce_mean(linear_rate)

	conv1_v = scnn_output[1]
	conv2_v = scnn_output[3] #was 2 before
	conv_pos = tf.nn.relu(conv1_v - thr_scnn)
	conv_neg = tf.nn.relu(-conv1_v - thr_scnn)

	if voltage_reg_method == 'avg_all':
		voltage_reg = tf.reduce_sum(tf.square(tf.reduce_mean(conv_pos, (0, 1))))
		voltage_reg += tf.reduce_sum(tf.square(tf.reduce_mean(conv_neg, (0, 1))))
	elif voltage_reg_method == 'avg_time':
		voltage_reg = tf.reduce_sum(tf.reduce_mean(tf.square(tf.reduce_mean(conv_pos, 1)), 0))
		voltage_reg += tf.reduce_sum(tf.reduce_mean(tf.square(tf.reduce_mean(conv_neg, 1)), 0))
	conv_pos = tf.nn.relu(conv2_v - thr_scnn)
	conv_neg = tf.nn.relu(-conv2_v - thr_scnn)
	if voltage_reg_method == 'avg_all':
		voltage_reg += tf.reduce_sum(tf.square(tf.reduce_mean(conv_pos, (0, 1))))
		voltage_reg += tf.reduce_sum(tf.square(tf.reduce_mean(conv_neg, (0, 1))))
	elif voltage_reg_method == 'avg_time':
		voltage_reg += tf.reduce_sum(tf.reduce_mean(tf.square(tf.reduce_mean(conv_pos, 1)), 0))
		voltage_reg += tf.reduce_sum(tf.reduce_mean(tf.square(tf.reduce_mean(conv_neg, 1)), 0))	
	
	
	reg_loss = rate_loss * rate_cost
	reg_loss += voltage_cost_rnn * voltage_reg_rnn
	reg_loss += voltage_cost_cnn * voltage_reg		
	
	return reg_loss
	
	
def calc_losses(logits, labels, scnn_output, core_output):
	
    loss_pred = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss_pred = tf.reduce_mean(loss_pred)

    entropy_loss = compute_entropy_loss(logits)
    loss_per_timestep = loss_pred + entropy_loss
    total_loss = tf.reduce_sum(loss_per_timestep)
    reg_loss = compute_reg_loss(scnn_output, core_output)
    total_loss += reg_loss

    # prediction = tf.argmax(logits, axis=2)
    # is_correct = tf.equal(labels, prediction)
    # is_correct_float = tf.cast(is_correct, dtype=tf.float32)
    # ler = 1. - tf.reduce_mean(ler)
    # decoded = prediction

    return total_loss

	# reg_factor_cnn = 1e8
	
	# # Calculate Loss		
	# pg_loss		 = compute_policy_gradient_loss(logits, action, advantages)
	# value_loss	 = compute_baseline_loss(advantages)
	# entropy_loss = compute_entropy_loss(logits)

	# loss_per_timestep = pg_loss + value_loss + entropy_loss
	# total_loss = tf.reduce_sum(loss_per_timestep)	
	
	# reg_loss = compute_reg_loss(scnn_output, core_output)	
	# total_loss += reg_loss

	# total_loss_for_cnn = tf.reduce_sum(pg_loss + value_loss) / reg_factor_cnn + reg_loss	
	
	# return total_loss, total_loss_for_cnn