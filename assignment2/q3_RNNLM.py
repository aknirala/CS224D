'''
Since we are training the language model, what exactly is expected?
I guess matrix L would be expected as if word vectors are already given, then model is already given.
But what about maxtrix W and I??
Why are they also part of language model?
Doesn't this mean that I can't use this model anywhere else????

'''
#A lot of code here is inspired/taken from: https://github.com/vijayvee/CS224d_Assignment_2_Solutions
#But few things have changed in TensorFlow so many things are rewritten.
# Also for sentence generation, the model pRmeters are taken and a new model has been cretaed.
# This is not an ideal way, but the intention was to make it working, asap

import getpass
import sys
import time
from q2_initialization import xavier_weight_init
import numpy as np
from copy import deepcopy

from utils import calculate_perplexity, get_ptb_dataset, Vocab
from utils import ptb_iterator, sample

import tensorflow as tf
#from tensorflow.python.ops.seq2seq import sequence_loss
#TF CHANGE: What is the non-legacy seq2seq
from tensorflow.contrib.legacy_seq2seq import sequence_loss
from model import LanguageModel





# Let's set the parameters of our model
# http://arxiv.org/pdf/1409.2329v4.pdf shows parameters that would achieve near
# SotA numbers

class Config(object):
  """Holds model hyperparams and data information.
  #
  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  batch_size = 64
  embed_size = 50
  hidden_size = 100
  num_steps = 10
  max_epochs = 16
  early_stopping = 2
  dropout = 0.9
  lr = 0.001

class RNNLM_Model(LanguageModel):
  #CLRF: What does vocab.encode do?
  '''This method gives self.
   vocab
   encoded_train/valid/test'''
  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    self.vocab = Vocab()
    self.vocab.construct(get_ptb_dataset('train'))
    self.encoded_train = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('train')],
        dtype=np.int32)
    self.encoded_valid = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('valid')],
        dtype=np.int32)
    self.encoded_test = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('test')],
        dtype=np.int32)
    if debug:
      num_debug = 1024
      self.encoded_train = self.encoded_train[:num_debug]
      self.encoded_valid = self.encoded_valid[:num_debug]
      self.encoded_test = self.encoded_test[:num_debug]

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible

    Adds following nodes to the computational graph.
    (When None is in a placeholder's shape, it's flexible)

    input_placeholder: Input placeholder tensor of shape
                       (None, num_steps), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, num_steps), type tf.float32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables

      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder

    (Don't change the variable names)
    """
    ### YOUR CODE HERE
    #raise NotImplementedError
    self.input_placeholder = tf.placeholder(tf.int32, shape = (None, self.config.num_steps))
    self.labels_placeholder = tf.placeholder(tf.float32, shape = (None, self.config.num_steps))
    self.dropout_placeholder = tf.placeholder(tf.float32)
    ### END YOUR CODE

  def add_embedding(self):
    """Add embedding layer.

    Hint: This layer should use the input_placeholder to index into the
          embedding.
    Hint: You might find tf.nn.embedding_lookup useful.
    Hint: You might find tf.split, tf.squeeze useful in constructing tensor inputs
    Hint: Check the last slide from the TensorFlow lecture.
    Hint: Here are the dimensions of the variables you will need to create:

      L: (len(self.vocab), embed_size)

    Returns:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    """

    # The embedding lookup is currently only implemented for the CPU
    with tf.device('/cpu:0'):
      ### YOUR CODE HERE
      L = tf.get_variable("L", (len(self.vocab), self.config.embed_size), initializer = tf.random_normal_initializer(-1, 1))  #Do I need to add trainable = True? What does it do?
      #This should be same as below line
      #L=tf.Variable(tf.random_uniform([len(self.vocab),self.config.embed_size],-1.0,1.0))
      inputs_L = tf.nn.embedding_lookup(L, self.input_placeholder)
      #TF CHANGE:::: For tf.split, arguments position has changed
      allSteps=tf.split(inputs_L,self.config.num_steps,1)#tf.split??????? Splits in 10 tensors
      inputs = [tf.squeeze(i, [1]) for i in allSteps]                     #tf.squeeze??????? removes 1 from [1]
      #raise NotImplementedError
      ### END YOUR CODE
      return inputs

  def add_projection(self, rnn_outputs):
    """Adds a projection layer.

    The projection layer transforms the hidden representation to a distribution
    over the vocabulary.

    Hint: Here are the dimensions of the variables you will need to
          create

          U:   (hidden_size, len(vocab))
          b_2: (len(vocab),)

    Args:
      rnn_outputs: List of length num_steps, each of whose elements should be
                   a tensor of shape (batch_size, embed_size).
    Returns:
      outputs: List of length num_steps, each a tensor of shape
               (batch_size, len(vocab)
    """
    ### YOUR CODE HERE
    #raise NotImplementedError
    #code here is inspired/taken from: https://github.com/vijayvee/CS224d_Assignment_2_Solutions
    shapeU = [self.config.hidden_size,  len(self.vocab)]
    shapeB2 = [len(self.vocab)]

    xavier_initializer = xavier_weight_init()

    outputs = []

    with tf.variable_scope("Project"):
      U = tf.get_variable("uWeights", shape = shapeU, initializer = xavier_initializer)
      b_2 = tf.get_variable("bias2", shape = shapeB2, initializer = xavier_initializer)
      #For all the rnn_outputs (num_steps) of them, calculate o/p via U and b_2
      for i in range(self.config.num_steps):
        outTensor = tf.matmul(rnn_outputs[i], U) + b_2
        outputs.append(outTensor)
    ### END YOUR CODE
    return outputs

  def add_loss_op(self, output):
    """Adds loss ops to the computational graph.

    Hint: Use tensorflow.python.ops.seq2seq.sequence_loss to implement sequence loss.

    Args:
      output: A tensor of shape (None, self.vocab)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    ### YOUR CODE HERE
    #raise NotImplementedError
    #code here is inspired/taken from: https://github.com/vijayvee/CS224d_Assignment_2_Solutions
    one = [tf.ones([self.config.batch_size*self.config.num_steps])]
    ce = sequence_loss([output], [tf.reshape(self.labels_placeholder, [-1])], one, len(self.vocab))

    tf.add_to_collection('ce_loss', ce)
    loss = tf.add_n(tf.get_collection('ce_loss'))
    ### END YOUR CODE
    return loss

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Hint: Use tf.train.AdamOptimizer for this model.
          Calling optimizer.minimize() will return a train_op object.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """
    ### YOUR CODE HERE
    #raise NotImplementedError
    return tf.train.AdamOptimizer().minimize(loss)
    ### END YOUR CODE
    #return train_op

  #We are building a tree over here. or The tree over here.
  def __init__(self, config):
    self.config = config
    self.load_data(debug=False)  #We'll get model.encoded_train/valid/test
    self.add_placeholders()           # (?, num_steps)
    self.inputs = self.add_embedding()  #Build the graph, to bring the inputs in vector form.
    #                                                                   inputs is a list (size num_steps:10) of tensors, each of dimension; ?, embed_size:50
    self.rnn_outputs = self.add_model(self.inputs) #It takes all the elements from inputs one by one, and produce an output. dim:
    self.outputs = self.add_projection(self.rnn_outputs)

    # We want to check how well we correctly predict the next word
    # We cast o to float64 as there are numerical issues at hand
    # (i.e. sum(output of softmax) = 1.00000298179 and not 1)
    self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
    # Reshape the output into len(vocab) sized chunks - the -1 says as many as
    # needed to evenly divide
    #This step seems new
    output = tf.reshape(tf.concat(1, self.outputs), [-1, len(self.vocab)])
    self.calculate_loss = self.add_loss_op(output)
    self.train_step = self.add_training_op(self.calculate_loss)


  def add_model(self, inputs):
    """Creates the RNN LM model.

    In the space provided below, you need to implement the equations for the
    RNN LM model. Note that you may NOT use built in rnn_cell functions from
    tensorflow.

    Hint: Use a zeros tensor of shape (batch_size, hidden_size) as
          initial state for the RNN. Add this to self as instance variable

          self.initial_state

          (Don't change variable name)
    Hint: Add the last RNN output to self as instance variable

          self.final_state

          (Don't change variable name)
    Hint: Make sure to apply dropout to the inputs and the outputs.
    Hint: Use a variable scope (e.g. "RNN") to define RNN variables.
    Hint: Perform an explicit for-loop over inputs. You can use
          scope.reuse_variables() to ensure that the weights used at each
          iteration (each time-step) are the same. (Make sure you don't call
          this for iteration 0 though or nothing will be initialized!)
    Hint: Here are the dimensions of the various variables you will need to
          create:

          H: (hidden_size, hidden_size)
          I: (embed_size, hidden_size)
          b_1: (hidden_size,)

    Args:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    Returns:
      outputs: List of length num_steps, each of whose elements should be
               a tensor of shape (batch_size, hidden_size)
    """
    ### YOUR CODE HERE
    #raise NotImplementedError
    #self.initial_state = tf.get_variable("initial_state", (1, self.config.hidden_size), initializer = tf.random_normal_initializer(-1.0, 1.0))
    self.initial_state = tf.zeros([self.config.batch_size, self.config.hidden_size])

    #h = tf.get_variable("h", (1, self.config.hidden_size), initializer = tf.constant_initializer(0.0))
    #H = tf.get_variable("H", (self.config.hidden_size, self.config.hidden_size), initializer = tf.random_normal_initializer(-1.0, 1.0))
    #I = tf.get_variable("I", (self.config.embed_size, self.config.hidden_size), initializer = tf.random_normal_initializer(-1.0, 1.0))
    #b1 = tf.get_variable("b1", (1, self.config.hidden_size), initializer = tf.random_normal_initializer(0.0))

    #tf.scope.reuse_variables()

    #h = tf.sigmoid(tf.matmul(tf.concat(H, I) , tf.concat(h, inputs) + b1 ))
    #rnn_outputs = tf
    ### END YOUR CODE
    #return h#rnn_outputs
    xavier_initializer=xavier_weight_init()
    shapeH=[self.config.hidden_size,self.config.hidden_size]
    shapeI=[self.config.embed_size,self.config.hidden_size]
    shapeB1=[self.config.hidden_size]
    outputs=[]
    with tf.variable_scope("InpDropout"):
    	inputs=[tf.nn.dropout(i,self.dropout_placeholder) for i in inputs]
    with tf.variable_scope("RNN") as scope:
    	H=tf.get_variable("H",shape=shapeH,initializer=xavier_initializer)
    	I=tf.get_variable("I",shape=shapeI,initializer=xavier_initializer)
    	b_1=tf.get_variable("b_1",shape=shapeB1,initializer=xavier_initializer)
    	outputs.append(self.initial_state)
    	for i in range(self.config.num_steps):
    		if(i!=0):
    			scope.reuse_variables()
    		hiddenTensor=tf.matmul(inputs[i],I) + tf.matmul(outputs[-1],H) + b_1
    		outputs.append(tf.nn.sigmoid(hiddenTensor))
    	self.final_state=outputs[-1]
    	rnn_outputs=outputs[1:]
    with tf.variable_scope("outDrop"):
    	rnn_outputs=[tf.nn.dropout(i,self.dropout_placeholder) for i in rnn_outputs]
    ### END YOUR CODE
    return rnn_outputs

    #                                model.encoded_train,                          = model.train_step)
  def run_epoch(self, session, data,                                          train_op=None, verbose=10):
    config = self.config
    dp = config.dropout
    if not train_op:
      train_op = tf.no_op()
      dp = 1
    total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
    total_loss = []
    state = self.initial_state.eval()
    for step, (x, y) in enumerate(
      ptb_iterator(data, config.batch_size, config.num_steps)):
      # We need to pass in the initial state and retrieve the final state to give
      # the RNN proper history
      feed = {self.input_placeholder: x,
              self.labels_placeholder: y,
              self.initial_state: state,
              self.dropout_placeholder: dp}
      loss, state, _ = session.run(
          [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
      total_loss.append(loss)
      if verbose and step % verbose == 0:
          sys.stdout.write('\r{} / {} : pp = {}'.format(
              step, total_steps, np.exp(np.mean(total_loss))))
          sys.stdout.flush()
    if verbose:
      sys.stdout.write('\r')
    return np.exp(np.mean(total_loss))



#This method has been rewritten.
#May be in future, I would come back and try to understand how to work with existing model.
#I am pretty sure, it is also effectively doing what I have done in the fucntion which I have implemented (below this function)
def generate_text_dep(session, model, config, starting_text='<eos>',
                  stop_length=100, stop_tokens=None, temp=1.0):
  """Generate text from the model.

  Hint: Create a feed-dictionary and use sess.run() to execute the model. Note
        that you will need to use model.initial_state as a key to feed_dict
  Hint: Fetch model.final_state and model.predictions[-1]. (You set
        model.final_state in add_model() and model.predictions is set in
        __init__)
  Hint: Store the outputs of running the model in local variables state and
        y_pred (used in the pre-implemented parts of this function.)

  Args:
    session: tf.Session() object
    model: Object of type RNNLM_Model
    config: A Config() object
    starting_text: Initial text passed to model.
  Returns:
    output: List of word idxs
  """
  state = model.initial_state.eval()
  # Imagine tokens as a batch size of one, length of len(tokens[0])
  tokens = [model.vocab.encode(word) for word in starting_text.split()]
  for i in xrange(stop_length):
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE
    next_word_idx = sample(y_pred[0], temperature=temp)
    tokens.append(next_word_idx)
    if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
      break
  output = [model.vocab.decode(word_idx) for word_idx in tokens]
  return output




def generate_text(session, model, config, starting_text='<eos>',
                  stop_length=10, stop_tokens=None, temp=1.0):
  '''
  In this function, I am extractintg the trained parameters from the existing graph and then
  build a model, actualy same model which was trained, and then computing the next word.

  I am simply assigning the hidden state to all zeros, then depending on the seed word, it woudl be brought to some state, from where, things woudl take place.
  '''
  #print dir(model)
  #print dir(session)
  #print session.graph
  #print dir(session.graph)
  #print session.graph.as_graph_element
  #print session.graph.get_all_collection_keys() #['ce_loss', 'trainable_variables', 'variables', 'train_op']
  for v in session.graph.get_collection('variables'):
    print v.name #First variable is L
    break

  #if True:
  #  return

  #print session.graph.unique_name()
  #print dir(session.graph.as_graph_element)
  with tf.variable_scope("RNNLM", reuse = True):
    #L = tf.get_variable(0)
    L = session.run(tf.cast(v, 'float64'))
    with tf.variable_scope("RNN", reuse = True) as scope:
      H=session.run(tf.cast(tf.get_variable("H"), 'float64'))#,shape=shapeH,initializer=xavier_initializer)
      I=session.run(tf.cast(tf.get_variable("I"), 'float64'))#,shape=shapeI,initializer=xavier_initializer)
      b_1=session.run(tf.cast(tf.get_variable("b_1"), 'float64'))#,shape=shapeB1,initializer=xavier_initializer)
      #print "Shape of parameters"
      #print H.shape
      #print I.shape
      #print b_1.shape
    with tf.variable_scope("Project"):
      U=session.run(tf.cast(tf.get_variable("uWeights"), 'float64'))
      b_2=session.run(tf.cast(tf.get_variable("bias2"), 'float64'))



  #Building our model here.
  #Create variables
  #print L.shape
  #print type(L[0])
  with tf.variable_scope("Pred") as scope:
    LP = tf.get_variable("L", initializer = L, dtype='float64')
    HP = tf.get_variable("H", initializer = H, dtype='float64')
    IP = tf.get_variable("I", initializer = I, dtype='float64')
    b_1P = tf.get_variable("b_1", initializer = b_1, dtype='float64')
    UP = tf.get_variable("U", initializer = U, dtype='float64')
    b_2P = tf.get_variable("b_2", initializer = b_2, dtype='float64')
    hP = tf.get_variable("h", initializer = np.zeros((1, 100)).view('float64'), dtype='float64')

    #First till prediction (but isn't it same?)
    x = tf.placeholder(tf.int32, shape=[1])

    emb = tf.nn.embedding_lookup(LP, x)
    hiddenTensor=tf.matmul(emb,IP) + tf.matmul(hP,HP) + b_1P
    scope.reuse_variables()
    hP = tf.nn.sigmoid(hiddenTensor)

    #Now, moving on to predicting the word.
    pred = tf.nn.softmax( tf.cast( tf.matmul(hP, UP) + b_2P, 'float64'))

    #Now, updating the value of hP
    tokens = [model.vocab.encode(word) for word in starting_text.split()]
    #h = np.zeros((1, 100)) #Should it be sth else?

    session.run(tf.initialize_all_variables())
    output = []
    for t in tokens:
      #h = np.dot(L[t].reshape(1,50) , I) + np.dot(h, H)
      #print h
      _, y_p = session.run([hP, pred], feed_dict={x:[t]})
      #print type(y_p)
      #print y_p.shape
      #print np.argmax(y_p)
      #print model.vocab.decode(np.argmax(y_p))

    for i in xrange(stop_length):
      nw = sample(y_p[0], temperature=temp)
      #print model.vocab.decode(nw)
      output.append(model.vocab.decode(nw))
      _, y_p = session.run([hP, pred], feed_dict={x:[nw]}) #np.argmax(y_p)
      if stop_tokens and model.vocab.decode(nw) in stop_tokens:
        break
    #
    return output

def generate_sentence(session, model, config, *args, **kwargs):
  """Convenice to generate a sentence from the model."""
  return generate_text(session, model, config, *args, stop_tokens=['<eos>'], **kwargs)

def test_RNNLM():
  config = Config()
  gen_config = deepcopy(config)
  gen_config.batch_size = gen_config.num_steps = 1

  # We create the training model and generative model
  with tf.variable_scope('RNNLM') as scope:
    model = RNNLM_Model(config)######HERE#############
    # This instructs gen_model to reuse the same variables as the model above
    #scope.reuse_variables()
    #Commenting it, as it is giving some error.
    #gen_model = RNNLM_Model(gen_config) #Why can't we simply pass the same config, why deep copy?

  init = tf.initialize_all_variables()
  saver = tf.train.Saver()
  #Why didn't they did: with tf.Graph().as_default():??????
  #Is some default, graph being used over here.
  with tf.Session() as session:
    best_val_pp = float('inf')
    best_val_epoch = 0

    session.run(init)
    for epoch in xrange(config.max_epochs):
      print 'Epoch {}'.format(epoch)
      start = time.time()
      ###
      train_pp = model.run_epoch(
          session, model.encoded_train,
          train_op=model.train_step) #train_step is optimizer.minimize(loss)
      valid_pp = model.run_epoch(session, model.encoded_valid)
      print 'Training perplexity: {}'.format(train_pp)
      print 'Validation perplexity: {}'.format(valid_pp)
      if valid_pp < best_val_pp:
        best_val_pp = valid_pp
        best_val_epoch = epoch
        saver.save(session, './ptb_rnnlm.weights')
      if epoch - best_val_epoch > config.early_stopping:
        break
      print 'Total time: {}'.format(time.time() - start)

    saver.restore(session, 'ptb_rnnlm.weights')
    test_pp = model.run_epoch(session, model.encoded_test)
    print '=-=' * 5
    print 'Test perplexity: {}'.format(test_pp)
    print '=-=' * 5
    starting_text = 'in palo alto'
    while starting_text:
      print ' '.join(generate_sentence(
          session, model, gen_config, starting_text=starting_text, temp=1.0))
      starting_text = raw_input('> ')

if __name__ == "__main__":
    test_RNNLM()
