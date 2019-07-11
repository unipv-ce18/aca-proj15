Advanced Computer Architectures
Forward and back propagation on artificial neural network
Project of: Prina Marco Ricci Stefano
Thanks to: Prof. Marco Ferretti Prof. Mirto Musci
1. INTRODUCTION<br>
In this project we implemented an algorithm that allows us to calculate a neural network.<br>
Our neural network, in particular, analyses a wine dataset, with relative properties and final classification.<br>
The language used for the implementation of the serial code is C, while the OpenMP library was used for the development of the parallelized code.<br>
The simulations were done on the Google platform, using different instances and Debian 9 was used as the operating system.<br>
RAM
4 GB
7 GB
15 GB
30 GB
60 GB
120 GB
vCPU
1
2
4
8
16
32
<br>
1.1 NEURAL NETWORK AND DATASET STRUCTURE<br>
The neural network is divided into two main components, nodes and weights.<br>
Our neural network is composed of:<br>
• Input Layer: 11 nodes corresponding to the 11 characteristics of a wine that we take into consideration.<br>
• Hidden Layer: the number of nodes can vary, but we have chosen to use 5 nodes.<br>
• Output Layer: single node that classifies the quality of the wine.<br>
In the implementation of our network we will also use the Bias nodes, one for the Input Layer and one for the Hidden Layer, both set to 1.<br>
The goal of the neural network is to tell us if a red wine, according to its characteristics is good or not.<br>
The dataset, which we used, was downloaded from "kaggle.com" and is called "Red Wine Quality", composed of 1599 wines, which we randomly divided into two subgroups:<br>
• The first, of 1279 samples, was used to train the net.<br>
• The second, consisting of the remaining 320 samples, was used to test the neural network and calculate the precision.<br>
Each single data (wine) is composed of:<br>
• 11 input values ("fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "Density", "pH", "sulphates", "alcohol").<br>
• 1 output value that represents the quality of the wine.<br>
Below we have decided to show how some characteristics of wine can influence the final quality. We can see for example how the quantity of “residual sugar” is almost uniform throughout the dataset, regardless of the final quality, on the contrary we can see how the “volatile acidity” changes considerably in relation to the final quality.<br>
The quality was initially a value in the interval [3,8], but due to the excessive imbalance of the dataset towards intermediate values, we decided to reduce the output to two values, "0" if the quality is between [3,5] and "1" if the quality is between [6,8].<br>
1.2 LOGISTIC REGRESSION<br>
Since this is a classification problem, we have used a non-linear regression model, specifically the Logistic Regression, which allows us to calculate the probability that a wine belongs to one class rather than another. The function used to calculate this probability is the Sigmoid Function:<br>
which allows us to keep the output value in the interval [0,1]. Therefore, based on the value returned by Sigmoid, a wine will be assigned class 0, if the value is less than 0.5, or class 1, if the value is greater than or equal to 0.5, these values predicted by the network will then be compared with the real output value to analyze the accuracy of the network.<br>
1.3 LOSS FUNCTION<br>
To train the network to give us increasingly precise results we introduce the concept of Loss Function, which allows us to calculate the error committed by the neural network in the prediction of output. The Loss Function must be strictly decreasing in a neural network.<br>
There are many ways to calculate the Loss Function, we have used the Cross-Entropy function:<br>
where 'N' stands for the number of samples, 'yn' is the real output while 'h(θ)n' is the output predicted by the network.<br>
Each time the entire dataset is traversed, the Cross-Entropy returns a value, which represents the average error committed by the neural network in calculating the outputs, compared to the real outputs.<br>
1.4 GRADIENT DESCENT<br>
Let's now analyze the method used to reduce the Loss Error: Gradient Descent. Gradient Descent is a first-order iterative optimization algorithm for finding the minimum of a function. The aim of this algorithm is to reduce the loss error at each iteration, by updating the variables, in this case the weights of the neural network. The updating of these weights takes place through the calculation of the derivative of the Loss Function multiplied by the Learning Rate, a variable that we can set as we like, which allows us to control the speed with which to reach the minimum.<br>
There are mainly three different implementations of the Gradient Descent: Stochastic, Batch and Mini-Batch.<br>
The Stochastic implementation, from which we started to develop our neural network, consists in updating the values of the weights after the analysis of each single sample, it is therefore necessary to randomize the dataset at each time in order to improve homogeneously the weight values. However, this implementation is not performing from the point of view of loss error, because it does not decrease in a linear way, but it was faster in the serial algorithm.<br>
The Batch implementation, on the other hand, consists in evaluating the entire dataset at each epoch, and through the average of the errors made on each single sample, updating the weights. In this case it is not necessary to randomize the dataset since the entire dataset is considered with each update. Using this implementation, the Loss Function has a strictly decreasing trend. We have chosen to use this implementation in our neural network, given the good results found. However, the serial algorithm using the entire Batch is slower than the Stochastic version but is more precise and lends itself better to parallelization.<br>
The implementation of the Mini-Batches consists of dividing the dataset into subgroups and updating the weights using the average of the error calculated on each data subgroup. This implementation is a middle ground between the Stochastic and the Batch one. In this case the error goes down more linearly than the stochastic implementation, but not strictly decreasing as in the case of the Batch.<br>
In the graph below, we can analyse the trend of the Loss Function, always decreasing. We also made a comparison considered 500 000 epochs and using different values for the Learning Rate (0.1; 0.3; 0.7; 1), and it is easy to see that as the Learning Rate increases, the function decreases faster.<br>
1.5 PRECISION<br>
The most important result for a neural network is the value of final precision. We have noticed that precision has never undergone variations from the use of the serial algorithm rather than the<br>

parallel one. The only variation he underwent was based on the various inputs that can be set at the start of the network: epochs, Hidden nodes and Learning Rate.<br>
In the graph shown, on a logarithmic scale, we used the values of the number of Hidden nodes and of the Learning Rate that we used for the entire network calculation, respectively 5 and 0.1.<br>
It can be seen that as the ages increase the accuracy increases, until it reaches a peak at 300,000 ages, then there is a slight decrease, probably due to overfitting.<br>
2. ANALYSIS OF THE SERIAL ALGORITHM<br>
Let us now analyse in detail how our serial algorithm works.<br>
For easier understanding, we divide it into three main phases:<br>
The first consists of loading the dataset and initializing nodes, variables and weights.<br>
The second is the real heart of network calculation, which is divided into two main phases: forward and back-propagation.<br>
The third phase consists in calculating the precision using the data that we have excluded from the training phase.<br>
2.1 INITIALIZATION<br>
After having initialized the number of neurons in the network, the number of epochs and the value of the Learning Rate, we are going to load our dataset.<br>
The dataset, as we have said, is divided into two files, "data.cvs" which contains the data to train the network, and "dataTest.csv" which contains the data to test the accuracy of the network. We have created a "struct data" where we can load all our dataset, this structure is divided into 2, input values and output values.<br>
Through the "readData" function we load the entire dataset on the "allData" and "allDataTest" structures. This function also inserts, for each sample, an initial value set to 1, which will be used to manage the Bias nodes.<br>
The code continues giving the possibility to set some values from the keyboard, and to choose whether to start only the serial, parallel or both algorithms.<br>
It continues with the initialization of the weights of the neural network both for the serial version and for the parallel version, using data saved for the first time randomly, which includes values between [-1.1].<br>
2.2 TRANING<br>
Now let's analyse in detail how the neural network is trained.<br>
After declaring the variables and matrices we are going to use, we set to 1 the value of the Bias node in the Hidden Layer.<br>
All our code is executed as many times as the number of epochs we set when the program started. For each epoch we set the "DeltaWeightIH" and "DeltaWeightHO" matrices to zero, which will keep track of the values necessary to update the weights calculated on each sample, and the "lossError" and "precision" variables that will calculate the final average of the error and of the precision calculated on the whole dataset.<br>
At this point we can begin to iterate over each sample of the dataset.<br>
2.2.1 FORWARD PROPAGATE<br>
The first step, also called Forward Propagate, first calculates the Hidden layer values, using the input values and relative weights, and normalizes these values using the Sigmoindal function, and then calculates the Output layer values, using the values just calculated in the Hidden nodes and the relative weights, always normalizing the values through the Sigmoidal function. Once these values have been calculated, it processes the error made using Cross-Entropy and updates the "lossError" variable. In the "DeltaO" matrix instead we save the difference between the expected output and the real output, necessary for the next phase.<br>
2.2.2 BACK PROPAGATION<br>
The second step consists of the Back-Propagate, which propagates the error calculated up to the Hidden layer, multiplying at first the "DeltaO" matrix for the respective "WeightHO", and the result is multiplied by the Sigmoindal derivative calculated using the values of the Hidden nodes, these values are saved in the "DeltaH" matrix.<br>
It continues with the calculation of the delta, derived from the error committed, necessary for the subsequent updating of the weights. These values are saved in the "DeltaWeightIH" and "DeltaWeightHO" matrices, iterating over the sample number.<br>
The last cycle simply deals with updating the value of the weights, for the next epoch, multiplying the average of the error, committed during the calculation of each sample, for the LearingRate.<br>
2.3 TESTING<br>
Now in the matrices "WeightIH" and "WeightHO" we have the last values of the weights calculated by the train algorithm. The final phase of our program consists in verifying how precise our network is in calculating the required output, using the data that was not used to train the network. In this phase we have implemented only the calculation related to Forward Propagate, because it is not necessary to propagate the error backwards, not having to further modify the weights. Then the test function takes care of comparing the predicted output from the network with the actual data output, and calculating the precision, i.e. the total number in which the network predicted the correct output on the total number of sample tests.<br>
3. A FIRST STUDY ON POSSIBLE PARRALLELS<br>
Looking at the project we decided to parallelize only the core of the program, that is the real neural network, ignoring all the various file reading and initialization functions necessary for use.<br>
Being passed from the stochastic to the batch model, it allowed us to parallelize the code more effectively, the process of activating neurons (forward propagation) and calculating the cost functions (back-propagation) can be done without having to respect an order in the samples and therefore simultaneously on different threads, unlike the stochastic model.<br>
We also judged parallelizable the setting of matrices to 0 every time it starts again a epoch.<br>
Instead, we realized that the necessary weight and deltaWeight calculation phase can be parallelized but we need to update a matrix using the "reduction (+: matrix [: r] [: c])", function present only in the most recent versions of openMP, but, remark that being an extremely fast calculation already in the serial line, this method even went to lengthen the times and the memory used as it creates a temporary matrix for each thread and then adds them together at the end.<br>
We then proceeded to the calculation of the theoretical speedup, to do this we measured in the serial code the percentage of time that the program uses on the parallelizable code compared to the total execution time in the neural network obtaining 69.4% on a network of 5 neurons for 1,000 000 epoch.<br>
percentage parallelizable code p = 0.694<br>
potential parallelization (# core) s = 1-2-4-8-16-32<br>
# Core
Theoretical Speedup
1
1
2
1,53
4
2,08
8
2,55
16
2,86
32
3,05
<br>
We expect the real speedup to be less than the theoretical one due to the parallelization set-up times.<br>
4. OpenMP PARALLEL IMPLEMENTATION<br>
To parallelize the code, we initially clashed with the setting of the matrices to 0 which consists of four "for" cycles nested two by two and not correlated with each other, so we decided to parallelize by specifying the characteristics collapse and, for the first, nowait:<br>
#pragma omp parallel
{
#pragma omp for collapse(2) nowait
for (i = 0; i <= numIn; i++) { /* initialize DeltaWeightIH */
for (j = 0; j <= numHid; j++) {
DeltaWeightIH[i][j] = 0.0;
}
}
#pragma omp for collapse(2)
for (j = 0; j <= numHid; j++) { /* initialize DeltaWeightHO */
for (k = 0; k <= numOut; k++) {
DeltaWeightHO[j][k] = 0.0;
}
}
}
The bulk of the parallelization gain comes from the "for" cycle on the samples for the calculation of forward and back propagation in which the order of computation of the samples does not influence the result so they can be done in parallel, but we had to set as private of variables used by each and previously declared including a vector (PartialDeltaH), we have also included in a negative reduction the lossError and in a positive the precision as they go respectively to subtract and add to themselves with each iteration.<br>
#pragma omp parallel for private(j, i, k, PartialDeltaH) \
reduction(-: lossError) reduction(+: precision)
for (int iteration = 1; iteration <= numSample; iteration++) {
…
lossError-=( … );
if (fabs(allData[iteration].out[k]-Output[iteration][k])< 0.5)
precision++;
}
<br>
For the next code we have, as already explained above, tried to use the reduction on the matrices ending up to lengthen the execution times of the parallelized code and we therefore decided not to use it in the final version.<br>
#pragma omp parallel
{
#pragma omp for reduction(+:WeightIH[:numIn+1][:numHid+1])\ collapse(2) nowait
for (i = 0; i <= numIn; i++) { /* update weights WeightIH */
for (j = 1; j <= numHid; j++) {
WeightIH[i][j] += … ;
}
}
#pragma omp for reduction(+:WeightHO[:numHid+1][:numOut+1]) collapse(2)
for (j = 0; j <= numHid; j++) { /* update weights WeightHO */
for (k = 1; k <= numOut; k++) {
WeightHO[j][k] += … ;
}
}
}
<br>
5. TEST AND DEBUGGING<br>
To make the program scalable, we used the malloc and realloc function to store the data read from files.<br>
In this way by simply setting the numIn and numHid variables we can in fact go to specify how many features will take from the file and how many initial weights must load.<br>
The loaded data can be separated by any delimiter, but it must be used the point and not a comma for decimals, also the program automatically adds a bias of 1 and continues to load until the file finishes and therefore regardless of the number of data.<br>
From many tests we have been able to ascertain that the values obtained are identical both to the final precision of the test performed and to the loss of the last epoch made independently from the use of the serial or parallelized program and therefore the latter only increases the performance of the program without to compromise the results.<br>
6. PERFORMANCE ANALYSIS<br>
For the analysis of the performances we have used different VMs with number of cores from 1 up to 32, we have also tried different configurations of the neural network as the number of epochs and the learning rate.<br>
As a main test we kept a high value of epochs (one million), 5 nodes in the hidden layer and a learning rate of 0.1 (which does not change the timing performance but only the final accuracy of the network).<br>
We immediately noticed that, as we expected, on a core there is not a decrease in the execution time between the serial and the parallel code and on two it is negligible, passing instead to 4 and then to 8 cores we notice each time a considerable increase in the parallel performance and then less and less for 16 and 32 cores probably due to the increase in the time required to set the parallelization and to the various fixed times.<br>
The time of the serial code is rightly not dependent on the number of cores instead.<br>
Going then to compare the theoretical speedup calculated above with that obtained from the various tests we can see a similar trend although, obviously, the real speedup is less than the ideal one but still manages to reach values close to 2 for computers with 16 or 32 core.<br>
We have also tried to change the number of epochs while maintaining the other factors constant, but we have not noticed significant fluctuations in speedup if not for very low number of epochs.<br>
7. CONCLUSION<br>
This project has given us the opportunity to confront ourselves with a new topic, that of neural networks. Having faced this problem without having basic knowledge and based on the results we obtained, such as accuracy over 76% and a real speed-up of almost 2, we can feel satisfied with our work, although we certainly realize that we have still a long way to go.<br>
