def cnn_2layer (n_inputs, n_outputs, height, width, channel):
	import tensorflow as tf
	X = tf.placeholder(tf.float32, shape=[None, n_inputs], name = "X")
	X_reshaped = tf.reshape(X, shape=[-1,height,width, channel])

	conv1 = tf.layers.conv2d(X_reshaped, filters=32, kernel_size = 3, strides = 1, padding="SAME", activation = tf.nn.relu, name="conv1")
	conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu, name="conv2")

	pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
	pool2_flat = tf.reshape(pool2, shape=[-1,64*50*50])

	fc1 = tf.layers.dense(pool2_flat, 64, activation = tf.nn.relu,name = "fc1")
	logits = tf.layers.dense(fc1, n_outputs, name = "output")
	Y_proba = tf.nn.softmax(logits, name="Y_proba")
	return logits, Y_proba, X
