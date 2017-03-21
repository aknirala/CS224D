import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    return x/np.sqrt(np.sum(np.multiply(x,x), axis = 1)).reshape(x.shape[0], 1)
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

    #return x

def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print x
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ""

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, assuming the softmax prediction function and cross
    # entropy loss.

    # Inputs:
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word
    # - outputVectors: "output" vectors (as rows) for all tokens
    # - dataset: needed for negative sampling, unused here.

    # Outputs:
    # - cost: cross entropy cost for the softmax word prediction
    # - gradPred: the gradient with respect to the predicted word
    #        vector
    # - grad: the gradient with respect to all the other word
    #        vectors

    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, using the negative sampling technique. K is the sample
    # size. You might want to use dataset.sampleTokenIdx() to sample
    # a random word index.
    #
    # Note: See test_word2vec below for dataset's initialization.
    #dataset.sampleTokenIdx = dummySampleTokenIdx
    #dataset.getRandomContext = getRandomContext



    #Here we have
    U = outputVectors
    #and
    u = U[predicted]
    v = target    #This is vc

    #First the positive part.
    sigma  = sigmoid(np.dot(u, v))
    J = np.log(sigma)

    vGrad = (1-sigma)*u
    UGrad = np.zeros(U.shape)
    UGrad[predicted] = (1-sigma)*v
    excludeIdx = [predicted, ]
    #Now the K negative parts
    for k in range(K):
        uSampIdx = dataset.sampleTokenIdx()
        while uSampIdx in excludeIdx:
            uSampIdx = dataset.sampleTokenIdx()
            #print 'collision'
        excludeIdx.append(uSampIdx)

        uSamp = U[uSampIdx]
        sigma = sigmoid(- np.dot(uSamp, v))
        J += np.log(sigma)
        vGrad -= (1-sigma)*uSamp
        UGrad[uSampIdx] -= (1-sigma)*v

    #Now we need to find gradient
    # Input/Output Specifications: same as softmaxCostAndGradient
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE
    cost = -J
    gradPred = -vGrad
    grad = -UGrad
    return cost, gradPred, grad

#        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    #Coding here for negSamplingCostAndGradient)
    if word2vecCostAndGradient == negSamplingCostAndGradient:
        J = 0
        Vgrad = np.zeros(inputVectors.shape)
        Ugrad = np.zeros(outputVectors.shape)
        for cw in contextWords:
            #negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
            j, gIn, gOut = negSamplingCostAndGradient(tokens[cw],  inputVectors[tokens[currentWord]],
                                                                                               outputVectors, dataset)
            J+= j
            Vgrad[tokens[currentWord]] += gIn
            Ugrad += gOut
        return J, Vgrad, Ugrad





    # Implement the skip-gram model in this function.

    # Inputs:
    # - currrentWord: a string of the current center word
    # - C: integer, context size
    # - contextWords: list of no more than 2*C strings, the context words
    # - tokens: a dictionary that maps words to their indices in
    #      the word vector list
    # - inputVectors: "input" word vectors (as rows) for all tokens
    # - outputVectors: "output" word vectors (as rows) for all tokens
    # - word2vecCostAndGradient: the cost and gradient function for
    #      a prediction vector given the target word vectors,
    #      could be one of the two cost functions you
    #      implemented above

    J = 0 #Calculating cost = u.v + exp...
    vc = inputVectors[tokens[currentWord]]

    #Second part
    for u in outputVectors:
        J += np.exp(np.dot(u, vc))

    J = len(contextWords) * np.log(J)  #Simply calculating the term 2*m * log of sum...

    #First part, calculating it later
    for uw in contextWords:
        u = outputVectors[tokens[uw]]
        J -= np.dot(u, vc)







    #Now implimenting gradients,
    V = inputVectors
    U = outputVectors

    #Vgrad = 0 for all components, but Vc
    Vgrad = np.zeros(inputVectors.shape)

    #Second term: 2m*sum(yw, uw)
    #With vc, we need to calculate |V| terms
    SMTerms = np.exp(np.dot(U, vc.T))
    denom = np.sum(SMTerms)
    Vgrad2 = len(contextWords)*np.dot(SMTerms, U)/denom

    #Now the first term, the negative one.
    Vgrad[tokens[currentWord]] = Vgrad2
    for uw in contextWords:
        u = U[tokens[uw]]
        Vgrad[tokens[currentWord]] -= u




    #Now the turn for Ugrad
    #First second term as it would be valid for all the outputs.
    #We can use denom and SMTerms from above
    Ugrad =   len(contextWords) * np.dot(SMTerms.reshape(len(tokens),1), vc.reshape(1, len(vc)))/denom   #Taking out len(contextWords), as for one word ..

    #Now we need to substract the first term for context words
    for uw in contextWords:
        Ugrad[tokens[uw] ]-=vc

    # Outputs:
    # - cost: the cost function value for the skip-gram model
    # - grad: the gradient with respect to the word vectors
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

    cost = J
    gradIn = Vgrad
    gradOut = Ugrad
    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.
    # Input/Output specifications: same as the skip-gram model
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #
    #################################################################

    #cost = 0
    #gradIn = np.zeros(inputVectors.shape)
    #gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    #raise NotImplementedError
    V = inputVectors
    U = outputVectors
    uc = U[tokens[currentWord]].reshape(1, U.shape[1])
    #Preparing v, from the context words.
    v = np.zeros(V.shape[1])
    for cw in contextWords:
        v += V[tokens[cw]]
    v /= len(contextWords)

    SMTerms = np.exp(np.dot(U, v.reshape(V.shape[1], 1)))
    denom = np.sum(SMTerms)
    J = np.log(denom)
    J -= np.dot(uc, v)


    #Now calculate the gradients
    VGrad = np.zeros(inputVectors.shape)
    #First the second term, as it would be common for all context words
    ct = (np.dot( SMTerms.reshape(1, U.shape[0]), U)/denom -uc)[0]/len(contextWords)
#    print ct.shape
#    print uc.shape
#    print VGrad[tokens[cw]].shape
    #Also substract the second term to get it completely
    for cw in contextWords:
        VGrad[tokens[cw]] += ct


    #Second term would be applicable for all u
    UGrad = np.dot(SMTerms.reshape( U.shape[0], 1), v.reshape(1, U.shape[1]))/denom
    #Now updating UGrad, just for the center word.
    UGrad[tokens[currentWord]] -= v
    ### END YOUR CODE
    cost = J
    gradIn = VGrad
    gradOut = UGrad
    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():      #Returns  a random number in [0, 4]
        return random.randint(0, 4)

    # dataset.sampleTokenIdx()
    def getRandomContext(C):             #Returns the random tokens.
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])       #Why are tokens in the form of a dictionary.
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)
    print "\n=CBOWWWWWW=="
    print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
