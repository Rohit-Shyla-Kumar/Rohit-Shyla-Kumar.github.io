"""
This script was written by SHYLA KUMAR Rohit for BScSCM FYP
Description - implements the agent class and soem helper function to create conv nets
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import itertools as it


#Some helper functions 

#function to create a single convolutional layer
def conv2d(input_, output_dim, 
        k_h=3, k_w=3, d_h=2, d_w=2, msra_coeff=1,
        name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=msra_coeff * get_stddev(input_, k_h, k_w)))
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        #add bias to the conv layer created inside
        return tf.nn.bias_add(tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME'), b)

#leaky relufor activation function 
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
    
#function to create one linear layer 
def linear(input_, output_size, name='linear', msra_coeff=1):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable("w", [shape[1], output_size], tf.float32,
                                tf.random_normal_initializer(stddev=msra_coeff * get_stddev(input_, 1, 1)))
        b = tf.get_variable("b", [output_size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, w) + b
    
#function to create a convolutional net
def conv_encoder(data, params, name, msra_coeff=1):
    layers = []                     #list of layers
    #nl is number of layers
    for nl, param in enumerate(params):
        if len(layers) == 0:        #if the net is empty, the input is the first layer
            curr_inp = data
        else:
            curr_inp = layers[-1]   #the input to each layer after the first is the previous layer
        #each layer is a conv layer with a lrelu activation function    
        layers.append(lrelu(conv2d(curr_inp, param['out_channels'], k_h=param['kernel'], k_w=param['kernel'], d_h=param['stride'], d_w=param['stride'], name=name + str(nl), msra_coeff=msra_coeff)))
        
    return layers[-1]

#function to create a fully connected net   
def fc_net(data, params, name, last_linear = False, return_layers = [-1], msra_coeff=1):
    layers = []                     #list of layers
    #nl is number of layers
    for nl, param in enumerate(params):
        #print("number of layers in fc net "+str(nl))
        if len(layers) == 0:        #if the net is empty, the input is the first layer        
            curr_inp = data
        else:
            curr_inp = layers[-1]   #the input to each layer after the first is the previous layer
        
        #each layer's name is the name of the net+the layer number
        if nl == len(params) - 1 and last_linear:
            #if its the last layer, connect it to the final output dimensions
            layers.append(linear(curr_inp, param['out_dims'], name=name + str(nl), msra_coeff=msra_coeff))
        else:
            #else, connect it to the corresponding output dimensions in the dict
            layers.append(lrelu(linear(curr_inp, param['out_dims'], name=name + str(nl), msra_coeff=msra_coeff)))
            
    if len(return_layers) == 1:
        return layers[return_layers[0]]
    else:
        return [layers[nl] for nl in return_layers]


#Agent class itself
class Agent:
    
    #constructor
    def __init__(self, sess, args):
        self.sess = sess
        
        # input data properties
        self.imgs_shape = args['imgs_shape']                    #(1,120,160)
        self.meas_shape = args['meas_shape']                    #3
        self.meas_for_net = args['meas_for_net']                            #range(3)
        self.meas_for_manual = args['meas_for_manual']                      #range(3,16)
        self.discrete_controls = args['discrete_controls']                  #[0...11]
        self.discrete_controls_manual = args['discrete_controls_manual']    #range(6,12)
        self.opposite_button_pairs = args['opposite_button_pairs']          #[(0,1),(2,3)]
        
        
        self.prepare_controls_and_actions()
        
        # preprocessing - the lambdas
        self.preprocess_input_images = args['preprocess_input_images']
        self.preprocess_input_measurements = args['preprocess_input_measurements']
        self.postprocess_predictions = args['postprocess_predictions']
        
        # net parameters
        self.conv_params = args['conv_params']                              #[(16,5,4), (32,3,2), (64,3,2), (128,3,2)]
        self.fc_img_params = args['fc_img_params']                          #[(128,)]
        self.fc_meas_params = args['fc_meas_params']                        #[(128,),(128,),(128,)]
        self.fc_joint_params = args['fc_joint_params']                      #[(256,),(256,),(-1,)]
        self.target_dim = args['target_dim']                                #18
            
        self.build_model()      
        
        
    def prepare_controls_and_actions(self):
        self.discrete_controls_to_net = np.array([i for i in range(len(self.discrete_controls)) if not i in self.discrete_controls_manual])
        #[0,1,2]
        self.num_manual_controls = len(self.discrete_controls_manual)
        #13
        
        self.net_discrete_actions = []      
        if not self.opposite_button_pairs:
            for perm in it.product([False, True], repeat=len(self.discrete_controls_to_net)):
                self.net_discrete_actions.append(list(perm))
        else:
            for perm in it.product([False, True], repeat=len(self.discrete_controls_to_net)):
            # remove actions where both opposite buttons are pressed 
                act = list(perm)
#                print(act)
                valid = True
                for bp in self.opposite_button_pairs:
                    if act[bp[0]] == act[bp[1]] == True:
                        valid=False
                if valid:
                    self.net_discrete_actions.append(act)
#            print(len(self.net_discrete_actions))
                    
        self.num_net_discrete_actions = len(self.net_discrete_actions)
        self.action_to_index = {tuple(val):ind for (ind,val) in enumerate(self.net_discrete_actions)}
        self.net_discrete_actions = np.array(self.net_discrete_actions)
        self.onehot_discrete_actions = np.eye(self.num_net_discrete_actions)
#        print(self.onehot_discrete_actions)
#        print(self.action_to_index)
       
        
        
    def preprocess_actions(self, acts):
        to_net_acts = acts[:,self.discrete_controls_to_net]
        return self.onehot_discrete_actions[np.array([self.action_to_index[tuple(act)] for act in to_net_acts.tolist()])]
        
    def load(self, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
    #            print("Model Loaded")
            return True
        else:
            return False

    
    
    
    def random_actions(self, num_samples):
        acts_net = np.random.randint(0, self.num_net_discrete_actions, num_samples)
        acts_manual = np.zeros((num_samples, self.num_manual_controls), dtype=np.bool)
        return self.postprocess_actions(acts_net, acts_manual)
        
    def make_net(self, input_images, input_measurements, input_actions, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        self.fc_val_params = np.copy(self.fc_joint_params)                  #[(256,),(256,),(-1,)]
        #changes it to [(256,),(256,),(18,)]
        self.fc_val_params['out_dims'][-1] = self.target_dim
        
        self.fc_adv_params = np.copy(self.fc_joint_params)                  #[(256,),(256,),(-1,)]  
        #changes it to [(256,),(256,),(648,)]
        self.fc_adv_params['out_dims'][-1] = len(self.net_discrete_actions) * self.target_dim
        
#        print(len(self.net_discrete_actions) * self.target_dim)
        
        p_img_conv = conv_encoder(input_images, self.conv_params, 'p_img_conv', msra_coeff=0.9)
        p_img_fc = fc_net(flatten(p_img_conv), self.fc_img_params, 'p_img_fc', msra_coeff=0.9)
        
        p_meas_fc = fc_net(input_measurements, self.fc_meas_params, 'p_meas_fc', msra_coeff=0.9)
        
#        print("flattend conv net "+str(flatten(p_img_conv)))
#        print("flattend conv net shape "+str((flatten(p_img_conv)).shape))
#        
#        
##        p_img_fc = tf.to_int32(p_img_fc)
#        print("image fc net "+str(p_img_fc))
#        print("image fc net shape"+str(p_img_fc.shape))
#        
#        
##        p_meas_fc = tf.to_int32(p_meas_fc)
#        print("measurements fc net "+str(p_meas_fc))
#        print("measurements fc net shape "+str(p_meas_fc.shape))
        
        data = tf.concat((p_img_fc,p_meas_fc),1)
#        print(data)
        
        p_val_fc = fc_net(data, self.fc_val_params, 'p_val_fc', last_linear=True, msra_coeff=0.9)
        p_adv_fc = fc_net(data, self.fc_adv_params, 'p_adv_fc', last_linear=True, msra_coeff=0.9)
        
        p_adv_fc_nomean = p_adv_fc - tf.reduce_mean(p_adv_fc, reduction_indices=1, keep_dims=True)  
        
        self.pred_all_nomean = tf.reshape(p_adv_fc_nomean, [-1, len(self.net_discrete_actions), self.target_dim])
        self.pred_all = self.pred_all_nomean + tf.reshape(p_val_fc, [-1, 1, self.target_dim])
        self.pred_relevant = tf.boolean_mask(self.pred_all, tf.cast(input_actions, tf.bool))
        
        
        
    def postprocess_actions(self, acts_net, acts_manual=[]):
        out_actions = np.zeros((acts_net.shape[0], len(self.discrete_controls)), dtype=np.int)
        out_actions[:,self.discrete_controls_to_net] = self.net_discrete_actions[acts_net]
        #print(acts_net, acts_manual, self.discrete_controls_to_net, out_actions)
        if len(acts_manual):
            out_actions[:,self.discrete_controls_manual] = acts_manual
        return out_actions
    
    
    
    def build_model(self):
        # prepare the data
        self.input_images = tf.placeholder(tf.float32, [None] + [self.imgs_shape[1], self.imgs_shape[2], self.imgs_shape[0]],
                                    name='input_images')
        self.input_measurements = tf.placeholder(tf.float32, [None] + list(self.meas_shape),
                                    name='input_measurements')
        self.input_actions = tf.placeholder(tf.float32, [None, self.num_net_discrete_actions],
                                    name='input_actions')
        
#        print("Input image shape "+str(self.input_images.shape))
#        print("Input measurements shape "+str(self.input_measurements.shape))
#        print("Input actions shape "+str(self.input_actions.shape))
        
        if self.preprocess_input_images:
            self.input_images_preprocessed = self.preprocess_input_images(self.input_images)
        if self.preprocess_input_measurements:
            self.input_measurements_preprocessed = self.preprocess_input_measurements(self.input_measurements)
        
        
        # make the actual net
        self.make_net(self.input_images_preprocessed, self.input_measurements_preprocessed, self.input_actions) 
        
        # make the saver, lr and param summaries
        self.saver = tf.train.Saver()

        tf.global_variables_initializer().run(session=self.sess)
    
    def act(self, state_imgs, state_meas, objective):
        return self.postprocess_actions(self.act_net(state_imgs, state_meas, objective), self.act_manual(state_meas)), None # last output should be predictions, but we omit these for now
        
    
    def act_manual(self, state_meas):
        if len(self.meas_for_manual) == 0:
            return []
        else:
            assert(len(self.meas_for_manual) == 13) # expected to be [AMMO2 AMMO3 AMMO4 AMMO5 AMMO6 AMMO7 WEAPON2 WEAPON3 WEAPON4 WEAPON5 WEAPON6 WEAPON7 SELECTED_WEAPON]
            assert(self.num_manual_controls == 6) # expected to be [SELECT_WEAPON2 SELECT_WEAPON3 SELECT_WEAPON4 SELECT_WEAPON5 SELECT_WEAPON6 SELECT_WEAPON7]
            
            curr_act = np.zeros((state_meas.shape[0],self.num_manual_controls), dtype=np.int)
            for ns in range(state_meas.shape[0]):
                # always pistol
                #if not state_meas[ns,self.meas_for_manual[12]] == 2:
                    #curr_act[ns, 0] = 1
                # best weapon
                curr_ammo = state_meas[ns,self.meas_for_manual[:6]]
                curr_weapons = state_meas[ns,self.meas_for_manual[6:12]]
                #print(curr_ammo,curr_weapons)
                available_weapons = np.logical_and(curr_ammo >= np.array([1,2,1,1,1,40]), curr_weapons)
                if any(available_weapons):
                    best_weapon = np.nonzero(available_weapons)[0][-1]
                    if not state_meas[ns,self.meas_for_manual[12]] == best_weapon+2:
                        curr_act[ns, best_weapon] = 1
            return curr_act
        
        
    def act_net(self, state_imgs, state_meas, objective):
        #Act given a state and objective
        predictions = self.sess.run(self.pred_all, feed_dict={self.input_images: state_imgs, 
                                                            self.input_measurements: state_meas[:,self.meas_for_net]})
            
        objectives = np.sum(predictions[:,:,objective[0]]*objective[1][None,None,:], axis=2)    
        curr_action = np.argmax(objectives, axis=1)
        return curr_action
    

        
#k_h and #k_w are the kernel height and width respectively 
get_stddev = lambda x, k_h, k_w: 1/np.sqrt(0.5*k_w*k_h*x.get_shape().as_list()[-1])



#flattens an input
def flatten(data):
    return tf.reshape(data, [-1, np.prod(data.get_shape().as_list()[1:])])

        
        
