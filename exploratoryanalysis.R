setwd("~/Documents/Master1_DataScience/1er QUADRI/Machine_Learning/Projects/Movie-rating-predictor")
# Data loading
data <- read.table("Data/train_user_movie_output_merge.csv", header=TRUE, sep=',')
attach(data)

# Data viewer
View(data)

data$rating <- as.numeric(as.character(data$rating))

# Statistical summaries of the variables
summary(data)
rating
boxplot(rating ~ gender)
boxplot(rating ~ occupation,las=2)
genre = colnames(data[3:21])
boxplot(rating ~ genre,las=2)

# boxplot(data[,2] ~ Classification, main='BMI (kg/m^2)')
boxplot(data[,3] ~ Classification, main='Glucose (mg/dL)')
boxplot(data[,4] ~ Classification, main='Insuline (uU/mL)')
boxplot(data[,5] ~ Classification, main='HOMA')
boxplot(data[,6] ~ Classification, main='Leptin (ng/mL)') 
boxplot(data[,7] ~ Classification, main='Adiponectin (ug/mL)')
boxplot(data[,8] ~ Classification, main='Resistin (ng/mL)')
boxplot(data[,9] ~ Classification, main='MCP.1 (pg/dL)')


features = c(1,2,22,23,24,25,26)

# Matrix of scatterplots of the quantitative features
pairs(data[,features])

# Graphics of the correlation matrix
c <- cor(data)
library(corrplot)
corrplot(c)