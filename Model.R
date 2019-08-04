################################
# Project: edX | Data Science: Capstone - MovieLens Project
# Author:  Kai Notté
# Date:    04.08.2019
# Version: 1.0
################################

################################
# GENERAL INFORMATION
# Please read related rmd-report for further information.
# The script follows each of the below mentioned steps
#
# 1.   Create edx set and validation set as provided in the edX Course Instruction
# 2.   Defining function RMSE to calculate the root mean square error
# 3.   Optimizing tuning parameter lambda on edx data set using user and movie biases
# 3.1. Split edx data set into train and test data set
# 3.2. Calculate RMSEs by iterating throught a sequence of lambdas
# 3.3. Define best tuning parameter lambda with the smallest RMSE
# 4.   Evaluate model by using the optimized tuning parameter lambda
###############################

###############################
# 1. Create edx and validation data set as provied by edx Course Instruction
###############################
# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###############################
# 2. Defining function RMSE to calculate the root mean square error
###############################

RMSE <- function(prediction, validation){
  sqrt(mean((validation - prediction)^2))
}

###############################
# 3. Optimizing tuning parameter lambda on edx data set using user and movie biases
###############################

###############################
# 3.1. Split edx data set into train and test data set
###############################

# Split edx data set into test data set (20% of edx data set) and train data set (80% of edx data set)
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train <- edx[-test_index,]
test_temp <- edx[test_index,]

# Make sure userId and movieId in test data set are also in train data set
test <- test_temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# Add rows removed from test data set back into train data set
test_removed <- anti_join(test_temp, test)
train <- rbind(train, test_removed)

###############################
# 3.2 Calculate RMSEs by iterating throught a sequence of lambdas
###############################

# Calculate mu: average rating of all movies
mu <- mean(train$rating)

# Define the sequenz of lambdas for the regulazation
lambdas <- seq(0, 10, 0.25)

# Iterate for each lambda: Calculate biases and evaluate prediction
rmse_byLamda <- sapply(lambdas, function(lambda){
  
  # Calculate the movie bias b_movie on the train data set
  b_movie <- train %>% 
    group_by(movieId) %>% 
    summarize(b_movie = sum(rating-mu) / (n() + lambda))
  
  # Calculate the user bias b_user_l on the train data set
  b_user <- train %>%
    left_join(b_movie, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_user = sum(rating - mu - b_movie) / (n() + lambda))
  
  # Calculate the predicted values for the test data set
  prediction <- test %>% 
    left_join(b_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    mutate(prediction = mu + b_movie + b_user) %>%
    pull(prediction)
  
  # Calculate the RMSE for each lambda
  return(RMSE(prediction, test$rating))
}) 

###############################
# 3.3. Define best tuning parameter lambda with the smallest RMSE
###############################
lambda <- lambdas[which.min(rmse_byLamda)]

###############################
# 4. Evaluate model by using the optimized tuning parameter lambda
###############################
# Calculate the movie bias b_movie on the edx data set with lambda = 4.75
b_movie <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_movie = sum(rating-mu) / (n() + lambda))

# Calculate the user bias b_user on the edx data set with lambda = 4.75
b_user <- edx %>%
  left_join(b_movie, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_user = sum(rating - mu - b_movie) / (n() + lambda))

# Calculate the predicted values for the validation data set
prediction <- validation %>% 
  left_join(b_movie, by = "movieId") %>%
  left_join(b_user, by = "userId") %>%
  mutate(prediction = mu + b_movie + b_user) %>%
  pull(prediction)

# Calculate the final RMSE
RMSE(prediction, validation$rating)
