import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
import io
from collections import Counter
from process import create_feature_sets_and_labels
train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
#from tensorflow.examples.tutorials.mnist import input_data

#423

#input > weights > hidden layer 1 (activation) > weights > hidden layer 2 (activation) > weights > output layer

#Compare output to intended output using cost or loss function(cross entropy)
#optimization function (optimizer) > minimize cost (AdamOptimizer)

#backpropagation (going backwards and manupulating the weights)

#feed forward + backpropagation = epoch

#mnist = input_data.read_data_sets("tmp/data/", one_hot = True)
lemmatizer = WordNetLemmatizer()
n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000
n_nodes_hl4 = 1000

n_classes = 2
batch_size = 10

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

def neural_network_model(data):
    
    hidden_1_layer = {'weights': tf.Variable(tf.truncated_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases': tf.Variable(tf.truncated_normal([n_nodes_hl1]))}
    
    hidden_2_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.truncated_normal([n_nodes_hl2]))}
    
    hidden_3_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.truncated_normal([n_nodes_hl3]))}
    
    hidden_4_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl3, n_nodes_hl4])),
                      'biases': tf.Variable(tf.truncated_normal([n_nodes_hl4]))}
    
    output_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl2, n_classes])),
                      'biases': tf.Variable(tf.truncated_normal([n_classes]))}
    
    # Rectilinear function relu (like the sigmoid function)

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )    
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 2
    
    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
            
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
                
            print('Epoch ', epoch, ' Completed out of ', hm_epochs, ' Loss: ', epoch_loss)
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy with our testing data: ', accuracy.eval({x: test_x, y: test_y}))

        with open('lexicon.pickle','rb') as f:
            lexicon = pickle.load(f)

        while 1:
            print("Enter your text (type end to stop)")
            take = input()
            if take == "end":
                break
            current_words = word_tokenize(take.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))

            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    # OR DO +=1, test both
                    features[index_value] += 1

            features = np.array(list(features))

            result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
            if result[0] == 0:
                print('Positive:', take)
            elif result[0] == 1:
                print('Negative:', take)


if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

train_neural_network(x)