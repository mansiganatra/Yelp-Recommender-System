# Yelp-Recommender-System

Implementation of collaborative filtering techniques incorporating ALS, LSH, Minhash and using Jaccard similarity measure. 

# Programming Environment
Python: 3.6  
Spark: 2.3.2  
Scala: 2.11  
spark.driver.memory :  4g  
spark.executor.memory: 4g  

# Hardware Specs:
Model: Lenovo Legion Y530 15  
OS: Windows 10, 64-bit  
Processor: Intel® Core™ i7-8750H CPU @ 2.20GHz x 6  
Memory: 16 GB  

# Generating dataset from Yelp Dataset
Generated the following two datasets from the original Yelp review dataset with some filters such as the condition: “state” == “CA”. We randomly took 60% of the data as the training dataset, 20% of the data as the validation dataset.  

a. yelp_train.csv: the training data, which only include the columns: user_id, business_id, and stars.  
b. yelp_val.csv: the validation data, which are in the same format as training data  

# Running the algorithm
# Task1: Jaccard based LSH
# Command to run:
spark-submit Mansi_Ganatra_task1.py <input_file_path> <output_file_path>  

Implemented the Locality Sensitive Hashing algorithm with Jaccard similarity using yelp_train.csv with focus on the “0 or 1” ratings rather than the actual ratings/stars from the users. Specifically, if a user has rated a business, the user’s contribution in the characteristic matrix is 1. If the user hasn’t rated the business, the contribution is 0. Identified similar businesses whose similarity >=0.5. The generated results are compared to the ground truth file pure_jaccard_similarity.csv.


# Task2: Recommendation System
# Command to run:
spark-submit Mansi_Ganatra_task2.py <train_filename> <tes_filename.py> <case_id> <output_filename>  

This implementation generates recommendations using:  
Case 1: Model-based CF recommendation system with Spark MLlib(using ALS implementation) 
Case 2: User-based CF recommendation system  
Case 3: Item-based CF recommendation system  
Case 4: Item-based CF recommendation system with Jaccard based LSH generated in task 1  

# Validation Set Baselines:
| Case | RMSE | Run Time(sec) |
| ------- | ------- | ------- |
| Case1 |	1.24 | 39 |
| Case2	| 1.15|	95 |
| Case3	| 1.15 | 109 |
| Case4 | 1.15 | 150 |

