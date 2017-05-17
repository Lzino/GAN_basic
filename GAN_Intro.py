# Our goal is
# Input z
# Output g like x


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import tensorflow as tf

sess = tf.InteractiveSession()

mu = 0.8
sigma = 0.1
num_bins = 100
# bins mean 샘플값을 셀 구간
num_samples = 5000

class GenerativeNetwork:
    dim_z = 1
    dim_g = 1

    def __init__(self):
        rand_uni = tf.random_uniform_initializer(-1e1,1e1)

        self.z_input = tf.placeholder(tf.float32, shape = [None,self.dim_z],name ="z-input")
        self.w0 = tf.Variable(rand_uni([self.dim_z, self.dim_g]))
        self.b0 = tf.Variable(rand_uni([self.dim_g]))

        self.g = tf.nn.sigmoid(tf.matmul(self.z_input,self.w0)+ self.b0)

    def generate(self, z_i):
        g_i = sess.run([self.g], feed_dict={self.z_input:z_i})
        return g_i[0]

class Discriminator:

    dim_x = 1
    dim_d = 1
    num_hidden_neurons = 10
    learning_rate = 1e-1

    x_input = tf.placeholder(tf.float32, shape=[None, dim_d], name ="Input_x")
    d_target = tf.placeholder(tf.float32, shape =[None, dim_d], name ="Target_d")

    rand_uni = tf.random_uniform_initializer(-1e1,1e1)

    w0 = tf.Variable(rand_uni([dim_x,num_hidden_neurons]))
    b0 = tf.Variable(rand_uni([num_hidden_neurons]))
    w1 = tf.Variable(rand_uni([num_hidden_neurons,dim_d]))
    b1 = tf.Variable(rand_uni([dim_d]))

    def __init__(self):
        self.d = self.getNetwork(self.x_input)
        self.loss = tf.losses.mean_squared_error(self.d, self.d_target)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def getNetwork(self, input):
        temp = tf.nn.tanh(tf.matmul(input,self.w0)+ self.b0)
        return tf.nn.sigmoid(tf.matmul(temp,self.w1) + self.b1)

    def discriminate(self, x_i):
        d_i = sess.run([self.d], feed_dict={self.x_input: x_i})
        return d_i[0]

    def train(self,x_i,d_i):
        error,_ =sess.run([self.loss, self.opt], feed_dict = {self.x_input: x_i, self.d_target:d_i})
        return error


def draw(x,z,g,D):
    # Drawing the histograms
    bins = np.linspace(0,1,num_bins)

    px, _ = np.histogram(x, bins = bins, density = True)
    pz, _ = np.histogram(z, bins = bins, density = True)
    pg, _ = np.histogram(g, bins = bins, density = True)
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
    v = np.linspace(0,1, len(px))

    v_i = np.reshape(v, (len(v), D.dim_x))
    db = D.discriminate(v_i)
    db = np.reshape(db, len(v))


    l = plt.plot(v,px,'b--',linewidth =1)
    l = plt.plot(v,pz, 'r--',linewidth =1)
    l = plt.plot(v,pg, 'g--', linewidth = 1)
    l = plt.plot(v,db, 'k--', linewidth = 1)
    plt.title('1D GAN Sample Test')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.show()
    plt.close()

def main():
    x = np.random.normal(mu,sigma,num_samples)
    #x means the real data made by normal distribution
    z = np.random.uniform(0, 1, num_samples)
    #z means the data made by uniform distribution
    #latent vector. the vector space's spot that represents images's feature
    g = np.ndarray(num_samples)

    #Define network
    G = GenerativeNetwork()
    D = Discriminator()
    #decision boundary #좋은 이미지인지 아닌지 0과 1로 판단

    #generate data
    tf.global_variables_initializer().run()

    x_i = np.reshape(x,(num_samples, D.dim_x))
    z_i = np.reshape(z,(num_samples, G.dim_z))
    g_i = G.generate(z_i)
    g = np.reshape(g_i,(num_samples))

    d_x_i = np.ndarray(shape = (num_samples, D.dim_x))
    d_x_i.fill(1.0)

    d_g_i = np.ndarray(shape = (num_samples, D.dim_x))
    d_g_i.fill(0.0)

#test-training
    for tr in range(0, 1000, 1):
        D.train(x_i,d_x_i)
        D.train(g_i,d_g_i)

        if tr % 100 == 0:
            print(D.train(x_i,d_x_i))
            print(D.train(g_i,d_g_i))

    draw(x,z,g,D)

    #GAN Algorithm

    #Generator optimizer
    D_from_g = D.getNetwork(G.g)

    loss_g = tf.reduce_mean(-tf.log(D_from_g))
    opt_g = tf.train.GradientDescentOptimizer(1e-3).minimize(loss_g)

    #Discriminator optimizer
    loss_d = tf.reduce_mean(-tf.log(D.d)-tf.log(1.0-D_from_g))
    opt_d =tf.train.GradientDescentOptimizer(1e-3).minimize(loss_d)

    #Train both
    frame_num = 0
    for tr in range(0, 10000, 1):
        # generate g from z again to respond the training of Generator
        g_i = G.generate(z_i)
        g = np.reshape(g_i, (num_samples))

        #train Discriminator from real/generated samples
        D.train(x_i,d_x_i)
        D.train(g_i,d_g_i)

        sess.run([loss_g, opt_g],feed_dict ={G.z_input:z_i})
        sess.run([loss_d, opt_d],feed_dict ={D.x_input:x_i,G.z_input:z_i})

        if tr % 1000 == 0:
            error_g,_ =sess.run([loss_g,opt_g], feed_dict ={G.z_input: z_i})
            error_d,_ =sess.run([loss_d,opt_d], feed_dict ={D.x_input: x_i, G.z_input: z_i})

        print(error_g, error_d)

    #generate g_is again after trainging Generator


    draw(x,z,g,D)

    print("frame_num", frame_num)
    # filename ="./capture/" + str(frame_num).zfill(5) + ".png"
    # plt.savefig(filename)
    # frame_num += 1
    # plt.close()



if __name__ == '__main__':
    main()
