# Movie Rating Prediction through Topic Modeling

The movie industry is growing in a rapid pace with billions of dollars of investments. Therefore, it is of utmost importance to predict the success of a particular film. In this paper, we build machine learning models to predict the success of a film through [**IMDb**](https://www.imdb.com/) ratings. Boosting methods, such as [**XGBoost**](https://xgboost.readthedocs.io/en/stable/) and [**CatBoost**](https://catboost.ai/) are employed, and the results of the models are compared using ROC-AUC metric. Besides including the given features in the models, we create an additional feature based on the topic of the movie synopsis using [**BERTopic**](https://github.com/MaartenGr/BERTopic) framework. The ROC-AUC scores of the models range from 0.79 to 0.82 with no additional power gained with the use of topics.

The analysis and scraped data can be found following [**this link**](https://drive.google.com/drive/folders/1u2cs1se4Gpzg7lNSrGHR-s5NnSy-rzDq?usp=sharing).

The paper and the summary of the research project are summarized in `Movie Rating Prediction through Topic Modeling` pdf file.