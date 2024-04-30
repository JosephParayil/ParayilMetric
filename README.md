# Introducing the Parayil Metric for obtaining Performance-Complexity graphs of machine learning models


A Performance-Complexity (PC) graph, or a Time-Complexity graph, depicts the relationship between the performance of an algorithm to the complexity of the task or environment. In computational sciences, these are very common and powerful for algorithm analysis, comparison, and optimization. 
<br><br>
<img src="https://adrianmejia.com/images/time-complexity-examples.png" alt="Time-Complexity graph" width="300">
<br><i>Figure 1: Time-complexity graph of common Big-O notation examples
<br>Image source: Adrian Mejia, adrianmejia.com/how-to-find-time-complexity-of-an-algorithm-code-big-o-notation/, accessed on 4/30/2024.</i>
<br>
<br>However, in the context of machine learning, such PC graphs depicting algorithm performance on learning a task over differing environment/task complexities are mostly absent, even though they would be highly valuable in the analysis of machine learning algorithms. The reason for this is due to the infeasibility of obtaining singular metrics for both machine learning model performance and environment complexity.
<br>In this research project, I overcome these fundamental challenges by inventing a novel mathematical method, involving what I have named the _Parayil Metric_. I am applying this method in the comparative analysis of the RL models Q-Learning (QL) and Deep Q-Learning (DQN) for their PC graphs. <br>

<h3>Getting a singular metric for machine learning performance</h3>

To construct a PC graph of a given machine learning model, I would have to be able to obtain a single numerical value that describes how well and fast the model learned at a certain environment. <br><br>
The data that we have, however, is a learning graph, a relationship between the timesteps of training and the performance over time. Over more timesteps or iterations of training, the model will generally become more adept. <br>
Converting a learning graph to a singular metric of how "good" the model trained is not straightforward. While there exists techniques such as finding the area under the graph, these are not well-suited methods for evaluating how good a learning graph is.
<br>
LEARNING GRPAH IMAGE
<br>
Additionally, obtaining a learning graph in the first place is not always straightforward. Due to epsilon decay, which applies in the algorithms of QL and DQN, doing a single run-through of the training process to obtain a learning graph in that way is not possible, because the number of total timesteps trained set to train for affects the rate of epsilon decay, which in turn skews the learning graph. To obtain a singular point of the learning graph, one would have to run the training process all the way until that point, and restart the process for another point, which quickly becomes overly time consuming.

<h3>Parayil Metric</h3
My solution was to kill these two birds with one stone, by inventing the Parayil Metric. The Parayil Metric aims to obtain a single numeric value to describe the nature of a given learning graph. 

How the Parayil Metric works is that it splits the y-values (performance, in terms of percentage win rate) into fixed intervals (ex: : 5%, 10% [...] 90% 95% 100%). For each of these y values, I find the corresponding x value (timestep). In other words, I find what is the timestep that yields an average win rate closest to the given y value. If the agent is not able to achieve the given win rate ever, an x value of infinity is assigned.  I then transform the x-values from timesteps to actual physical time. So I would run tests determining on average how much time does it take to run the given amount of timesteps of training, running plenty of trials to account for randomness and differences in computational resources.  For each of these transformed x-y pairs, I find the slopes by dividing the y over the x. Finally, the Parayil Metric would then be the average of all these slope values. 
<br><br>
Such a metric takes into account learning over time and is appropriately sensitive to performance limits that the algorithm may encounter in the environment. The Parayil Metric of one algorithm can then be compared with that of another algorithm, to find the comparative performance on a given environment. Additionally, it is applied in constructing the performance-complexity graph. 
<br><br>
Another benefit of using the Parayil Metric is that I only need to run timestep tests specifically for the y-value intervals, instead of scanning the entire timestep space (which would take astronomically long due to the epsilon decay problem). To find the corresponding x value of each y value, I use the binary search algorithm, which is normally used for searching for items in a list, but can be applied in this case quite well.
<br><br><br>
<img src="ParayilMetric.png" alt="Parayil Metric" width="500">

                    


<br><br>
 
<br>
<h3>Getting a singular metric for environment complexity</h3>

