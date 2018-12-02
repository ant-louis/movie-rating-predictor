setwd("~/Documents/Master1_DataScience/1er QUADRI/Machine_Learning/Projects/Movie-rating-predictor")
# Data loading
data <- read.table("Data/train_user_movie_output_merge.csv", header=TRUE, sep=',')
attach(data)
detach(data)
# Data viewer
View(data)

data$rating <- as.numeric(as.character(data$rating))

# Statistical summaries of the variables
summary(data)


boxplot(rating ~ gender)
boxplot(rating ~ occupation,las=2)

features = c(1,2,22,23,24,25,26)

# Matrix of scatterplots of the quantitative features
pairs(data[,features])

