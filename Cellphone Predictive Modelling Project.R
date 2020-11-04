1.#EDA#
##Importing the Libraries##
library(readr)
library(readxl)
library(ggplot2)
library(gridExtra)
library(DataExplorer)
library(dplyr)
library(corrplot)
library(car)
library(caret)
library(lattice)
library(e1071)
library(caTools)
library(ROCR)
library(pROC)
library(blorr)
library(kableExtra)
######################

setwd("F:/GREAT LEARNING/PREDICTIVE MODELLING/Project - Cellphone/GL- Solution")
getwd()
##Importing the dataset##
cell=read.csv("Cellphone.csv",header = TRUE)
##Checking the dimensions of dataset##
dim(cell)
## Converting Churn , contract Renewal and Data plan as factored variables as they have valve as 0 or 1
cell$Churn=as.factor(cell$Churn)
cell$ContractRenewal=as.factor(cell$ContractRenewal)
cell$DataPlan=as.factor(cell$DataPlan)
##Structure of the cell Dataset##
str(cell)
## Summary of cell Dataset##
summary(cell)

##Exploratory Data Analysis##
##Introductory plot of dataset##
plot_intro(cell)
##Histogram of Variables##
plot_histogram(cell,geom_histogram_args = list(fill="blue"),
               theme_config = list(axis.line=element_line(size=1,colour="green"),
                                   strip.background=element_rect(color="red",fill="yellow")))
##checking the distribution of variables##
##Density Plots of variables##
plot_density(cell,geom_density_args = list(fill="gold",alpha=0.4))

##Checking of Outliers##
##Checking outliers with respect to churn responsible variable
plot_boxplot(cell,by ="Churn", geom_boxplot_args = list("outlier.color" = "red", fill="blue"))

##Bivariate Analysis##
p1 = ggplot(cell, aes(AccountWeeks, fill=Churn)) + geom_density(alpha=0.4)
p2 = ggplot(cell, aes(MonthlyCharge, fill=Churn)) + geom_density(alpha=0.4)
p3 = ggplot(cell, aes(CustServCalls, fill=Churn))+geom_bar(position = "dodge")
p4 = ggplot(cell, aes(RoamMins, fill=Churn)) + geom_histogram(bins = 50, color=c("red")) 
grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)

##Check for Multicollinearity##
cell.numeric=cell%>% select_if(is.numeric)
a=round(cor(cell.numeric),2)
corrplot(a)

## Classifier Models ##
##Splitting dataset -Train and Test

## splitting dataset into train test in the ratio of 70:30 %
set.seed(233)
split = createDataPartition(cell$Churn , p=0.7, list = FALSE)

train.cell = cell[split,]
test.cell = cell[-split,]

##checking dimensions of train and test splits of dataset
dim(train.cell)
dim(test.cell)

##Matrix for check of split of Response var in Train and Test Datasets##
table(train.cell$Churn)
table(test.cell$Churn)

##KNN-Nearest Neighbour Classfier##

trctl = trainControl(method = "repeatedcv", number = 10, repeats = 3)

set.seed(1111)
knn.fit = train(Churn~., data = train.cell, method="knn",
                trControl= trctl, preProcess = c("center", "scale"),
                tuneLength= 10)
knn.fit

##Interpretation of K-NN##
knn.pred=predict(knn.fit,test.cell)
mean(knn.pred==test.cell$Churn)

##Confusion Matrix for K-NN##
knn.CM=confusionMatrix(knn.pred,test.cell$Churn,positive = "1")
knn.CM

##Naive Bayes classifier##
NB.fit=naiveBayes(Churn~.,data = train.cell)
NB.fit

##Interpretation of Naive Bayes##
NB.pred=predict(NB.fit,test.cell,type = "class")
mean(NB.pred==test.cell$Churn)

NB.CM=confusionMatrix(NB.pred,test.cell$Churn,positive = "1")
NB.CM

##Logistic Regression Classifier##

##Running a Logit R through glm##
logitR.fit=glm(Churn~.,data = train.cell,family="binomial")
summary(logitR.fit)

##Checking for variance inflation Factor##
vif(logitR.fit)

##chi square test to check the significant predictors with varying sig levels##
anova(logitR.fit, test = "Chisq")

##Interpretation of Logit Regression##
logitR.pred = predict(logitR.fit, newdata = test.cell, type = "response")

logitR.predicted = ifelse(logitR.pred > 0.5 , 1, 0)
logitR.predF = factor(logitR.predicted, levels = c(0,1))

mean(logitR.predF == test.cell$Churn)

logitR.CM=confusionMatrix(logitR.predF,test.cell$Churn,positive = "1")
##Confusion Matrix for LOgitR model##
logitR.CM

##ROC Curve for LR Model##
#AUC or Area under the curve is 78% ie dataset has 78.6% concordant pairs#

ROCRpred = prediction(logitR.pred, test.cell$Churn)
AUC=as.numeric(performance(ROCRpred, "auc")@y.values)
## Area under the curve for LR model
AUC

##ROC Curve for the model##
perf=performance(ROCRpred,"tpr","fpr")
plot(perf,col="black",lty=2,lwd=2,colorize=T,main="ROC curve Admissions",xlab="specificity",
     ylab="sensitivity")
abline(0,1)

##KS Curve for the model##
ks=blr_gains_table(logitR.fit)
blr_ks_chart(ks,title="ks chart",
             yaxis_title = "",xaxis_title = "cumulative population %",
             ks_line_color = "black")
          
                                ####THANK YOU####


