# ml-recommendation-system
### Machine Learning Group Project @Georgia Tech

#### Topic: Recommendation system for fashion products


To reproduce the result:

1. clone this repository
  ```
  git clone https://github.gatech.edu/lyang417/ml-recommendation-system.git
  ```

2. Install the dataset from the kaggle competition: 
  ```
  kaggle competitions download -c h-and-m-personalized-fashion-recommendations
  ```
  
3. make sure everything installed from kaggle is under ```data/``` directory in your working directory


4. For preprocessing: Create a txt file (i.e image_list.txt)  that contains paths of the images (they are under data/images) you want to preprocess line by   line and run
  ```
  python3 preprocess.py -i image_list.txt
  ```

5. Train the model: 
  ```
  python3 -u main.py -m autoencoder_paper -b 16 -e 30 -l 0.01
  ```
