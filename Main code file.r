#' ---
#' title: next generation model for temperature shock experiment
#' author: Nicholas M. Fountain-Jones, 
#' 

#' We start the pipeline by clearing the work space and loading all of the required packages.
#' 
## ---- message=F, warning=F-----------------------------------------------
rm(list = ls())
library(memisc)
library("randomForest");library("caret");library("pROC");library("ROCR");library("plyr");library("missForest")
library("gbm");library("pdp");library("ggplot2"); library("iml");library("dplyr") library("kernlab")


#data <- read.csv("Thermal_shock.csv")
data <- read.table(file="Data7.txt", header=T, sep="\t")
data0 <- data
data$Rel_Hum_pc <- NULL
## ---- data exploration-----------------------------------------------

library(DataExplorer)
plot_missing(data) ## Are there missing values, and what is the missing data profile? None in this case
plot_bar(data) ## How does the categorical frequency for each discrete variable look like?
plot_histogram(data) ## What is the distribution of each continuous variable?

names(data)
#  [1] "Species"               "Temperature"           "Exposure_hs"          
#  [4] "Rel_Hum_pc"            "Stage"                 "Sex"                  
#  [7] "Food_regime"           "Food_source"           "N"                    
# [10] "Temp_2nd_shock"        "Exposure_2nd_shock_hs" "Interv_bw_shocks_hs"  
# [13] "Mort_pc"               "Molting_rate_pc" 

colnames(data)[2] <- 'temp' #there was a mistake here (changed species to temp)


str(data)

#### We select only the "apparently" worth while variables
## The original dat was kept as a duplicate in data0

data$Molting_rate_pc <- NULL
data$Food_source <- NULL
data$mort <- NULL
data$Temp_2nd_shock <- NULL
data$Exposure_2nd_shock_hs <- NULL
data$Interv_bw_shocks_hs <- NULL


names(data)


##### We add another dataset for mortality pooling all species
##### Decision after checking (see below "plot(imp_mort)+ theme_bw()#plot results")
##### that showed that species have almost no importance


#as we are interested in mortality....
plot_boxplot(data_mort, by = "Mort_pc") 
plot_scatterplot(data_molt, by = "Molting_rate_pc")

#check correlations (unlikely in this case but good practice)
plot_correlation(data)
#removed food regime
data$Food_regime <- NULL

#Can impute N with missingForests
library(missForest)
set.seed(135) #this is stochastic algorithm so set seed to ensure results can be fully replicated.
data <- missForest(data, variablewise = TRUE)
data$OOBerror #checks error rate for missing data impuation for each variable.
# #make a dataframe again.
data <- as.data.frame(data$ximp)

#high MSE for N (113 units)
data$N <- NULL
data$Sex <- NULL
str(data)

#categorical predictors here means we need dummy variables

#Dummies <- dummyVars( Mort_pc~., data = data)


#DataA <- data.frame(predict(Dummies, newdata = data))

#data1 <- cbind(data$Mort_pc, DataA)
#colnames(data1)[1] <- 'mort'
# why was Stage removed? Correlation is high but maybe worth keeping...
#DataA$Stage <- NULL

#no need for dummies for random forests/GBM


#-----------

#split into testing and training data
## 75% of the sample size
smp_size_mort <- floor(0.75 * nrow(data))


## set the seed to make your partition reproducible
set.seed(123)
train_ind_mort <- sample(seq_len(nrow(data)), size = smp_size_mort)
train_ind_mort <- sample(seq_len(nrow(data)), size = smp_size_mort)

train_mort <- data[train_ind_mort, ]
test_mort <- data[-train_ind_mort, ]


## ------------------------------------------------------------------------
## --------------------Gradient Boosting------------------------------
## ------------------------------------------------------------------------


set.seed(123)
fitControl <- trainControl(method = 'repeatedcv', number=10, repeats=10) #10 fold CV this time
gbmGrid <- expand.grid(n.trees = seq(50, 4000,300), interaction.depth = c(1,5,9), shrinkage= c(0.01, 0.001), n.minobsinnode=5) #n.minobsinnode=5 maybe best for regression
#how many models to test
nrow(gbmGrid)


mod_gbm_mort <- train(
  Mort_pc ~ .,
  data = data,
  method = "gbm",
  trControl = fitControl,
  tuneGrid = gbmGrid, verbose=F)

plot(mod_gbm_mort)
mod_gbm_mort
summary(mod_gbm_mort)
#check perfomance
getTrainPerf(mod_gbm_mort)
pred_mort <- predict(mod_gbm_mort,test_mort )

#better performance than SVM RMSE = 18




## ------------------------------------------------------------------------
## --------------------Support Vector machine------------------------------
## ------------------------------------------------------------------------

#The gamma parameter in the RBF kernel determines the reach of a single training instance. If the value of Gamma is low,
#then every training instance will have a far reach. Conversely, high values of gamma mean that training instances will have a close reach. So, with a high value of gamma, the SVM decision boundary 
#will simply be dependent on just the points that are closest to the decision boundary, effectively ignoring points that are farther away. In comparison, a low value of gamma will result in a decision boundary that will consider points that are further from it.
#As a result, high values of gamma typically produce highly flexed decision boundaries, and low values of gamma often results in a decision boundary that is more linear.
set.seed(123)

#### First for mortality
mod_svm_mort <- train(
  mort ~ .,
  data = data1,
  method = "svmRadial",
  trControl = trainControl(method = "cv", number = 5), #5 fold cross validation
  tuneGrid = expand.grid(sigma= 2^c(-25, -20, -15,-10, -5, 0), C= 2^c(0:5)))

plot(mod_svm_mort)




#check perfomance
getTrainPerf(mod_svm_mort)

#best model automatically included

## ------------------------------------------------------------------------
## --------------------Linnear model------------------------------
## ------------------------------------------------------------------------

#### First for mortality
mod_lm_mort <- train(mort  ~ .,
                 data = data1,
                 method  = "lm",
                 trControl = trainControl(method = "cv", number = 5), #5 fold cross validation
                 tuneGrid  = expand.grid(intercept = FALSE))

getTrainPerf(mod_lm_mort)


#lm seems to be the best model - this RMSE means that the average error in preediction are around 14 mortality events

# To visualize feature performance we use the model agnostic 'model class reliance' method (see main text). Feature importance can be interpreted as the amount (or factor) by which model error is increased by removing this feature compared to the original model error. All plots use ggplot2 graphics and are based on code by Molnar (2018).

## ------------------------------------------------------------------------
## --------------------Bayesian GLMM------------------------------
## ------------------------------------------------------------------------
library(arm)

#### First for mortality

mod_bglm_mort <- train(mort  ~ .,
                data = data1,
                method  = "bayesglm",
                trControl = trainControl(method = "cv", number = 5))

getTrainPerf(mod_bglm_mort)

#not quite as good as the others



## ------------------------------------------------------------------------
X_mort <-data[-which(names(data) == "Mort_pc")] #load data again for the visualization
Y_mort <- data$Mort_pc

str(data)
# create the iml object

library(iml)
library(devtools)

#####################################################################
######################## First for mortality ########################
#####################################################################

modint_mort <-Predictor$new(mod_gbm_mort, data = data) #create predictor object. Add your model object name here (GLM, RF, GBM or SVM)
set.seed(123)
imp_mort <-FeatureImp$new(modint_mort, loss = "mse", compare='ratio', n.repetitions = 20) #25 is the maximum 
imp.dat_mort<- imp_mort$results
plot(imp_mort)+ theme_bw()#plot results

top2_mort<- imp.dat_mort$feature[1:4]#n = number of predictors you want to display


ice_curves_mort <- lapply(top2_mort, FUN = function(x) {
  cice <- partial(mod_gbm_mort #have to change this
                  , pred.var = x, center = TRUE, ice = TRUE,
                  prob = T) #note that we center values in the plotso these are centered ICE plots (cICE)
  autoplot(cice, rug = TRUE, train = data, alpha = 0.1) +
    theme_bw() +
    ylab("c-ICE")
})

Categorical <- "Species"
ice_curves_cat <- lapply(Categorical, FUN = function(x) {
  ice <- partial(mod_gbm_mort, pred.var =  'Species', ice = TRUE, center = FALSE,
                 prob = T)
  ggplot(ice, rug=T, train = data, aes(x=Species, y = yhat, group = Species)) +
    geom_boxplot()+theme_bw() 
})
#put them together
grid.arrange(grobs = c(ice_curves_mort, ice_curves_cat), ncol = 2)

#put them together
grid.arrange(grobs = c(ice_curves_mort), ncol = 2)

#' ###4.3 Interactions using Friedman's H index
#' 
#' Calculating Friedman's H index provides a robust way to assess the importance of interactions in shaping risk across models. The interactions identified can then be visualized using PD plots.
#' 
## ------------------------------------------------------------------------
set.seed(345)

interact <- Interaction$new(modint_mort)
plot(interact)+ theme_bw()

#' 
#' This plot shows that Age_exposed has the highest H score and is involved in the most interactions with other features in shaping parvovirus risk. We can interrogate these interactions and plot the strongest identified with the following code:
#' 
## ---- warning=F----------------------------------------------------------
interact1 <- Interaction$new(modint_mort, feature = "Exposure_hs")

plot(interact1)+theme_bw()

pdp.obj <-  FeatureEffect$new(modint_mort, feature = c("Exposure_hs","Species"), method='pdp')


plot(pdp.obj)+ scale_fill_gradient(low = "white", high = "red")+ theme_bw()

pdp.obj1 <-  FeatureEffect$new(modint_mort, feature = c("Species","temp"), method='pdp')


plot(pdp.obj1)+ scale_fill_gradient(low = "white", high = "red")+ theme_bw()

pdp.obj <-  FeatureEffect$new(modint_mort, feature = c("Stage","temp"), method='pdp')


plot(pdp.obj)+ scale_fill_gradient(low = "white", high = "red")+ theme_bw()
