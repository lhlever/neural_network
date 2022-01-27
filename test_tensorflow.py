import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
coefficient=np.array([[1.],[-60.],[900.]])
w=tf.Variable(0,dtype=tf.float32)
x=tf.placeholder(tf.float32,[3,1])
# cost=tf.add(tf.add(w**2,tf.multiply(-10.0,w)),25.0)
cost=x[0][0]*w**2+x[1][0]*w+x[2][0]
tf.disable_eager_execution()
train=tf.train.GradientDescentOptimizer(0.01).minimize(cost,var_list=w)
init=tf.global_variables_initializer()
# tf.compat.v1.disable_eager_execution()
# session=tf.Session()
with tf.Session() as session:
    session.run(init)
    print(session.run(w))
    session.run(train,feed_dict={x:coefficient})
    print(session.run(w))
    for i in range(1000):
        session.run(train,feed_dict={x:coefficient})
    print(session.run(w))
