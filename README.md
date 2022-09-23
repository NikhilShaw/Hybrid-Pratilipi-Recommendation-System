# Hybrid-Pratilipi-Recommendation-System

Different approaches of recommending Pratilipis. <br />
Combination of user-based collabarative filtering, content-based collabarative filtering and most popular Pratilipis. <br />

Installation <br />
````
$ cd.. content_based_recommendation_system
$ pip install -r requirement.txt
````

Running instructions <br />
1. To run content based recommendation system <br />
````
Place metadata.csv and user-interactions,csv in content_based_recommendation_system/data folder
$ cd.. content_based_recommendation_system
$ python main.py
````

2. To run collabarative filtering based recommendation system <br />
````
Place metadata.csv and user-interactions,csv in collaborative_filtering_based_recommendation/data folder
$ cd collaborative_filtering_based_recommendation
$ python main.py 
````

Config: <br />
````
Both content_based_recommendation_system and collaborative_filtering_based_recommendation have config which can limit the train and test data
````

Question and answers can be found in "Question and Answers.docx"

