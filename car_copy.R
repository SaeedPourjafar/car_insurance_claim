library(VIM) # Visualizing the missing values
library(mice) # Imputing the missing values
library(lattice) # For densityplot
library(dplyr)
library(readr)
library(ggplot2)
library(ggcorrplot)
library(cowplot) # For plot_grid
library(corrplot)
library(stringr)
library(caret)
library(ROCR)
library(broom) # To augment data
library(verification) # roc.area and roc.plot
library(arm) # bayesglm and binnedplot
library(bestNormalize) # yeojohnson transformation

options(scipen=999)
setwd("C:\\uw\\DSBA 2\\Machine Learning 1\\Project")
car <- read_csv("car.csv")
glimpse(car)
dim(car)
car$ID <- NULL # Removing ID

# Removing the dollar sign and comma, then convert it to numeric 
car$INCOME[!is.na(car$INCOME)] <- as.numeric(gsub('[$,]','',
                                          car$INCOME[!is.na(car$INCOME)]))
car$INCOME <- as.numeric(car$INCOME)
car$HOME_VAL[!is.na(car$HOME_VAL)] <- as.numeric(gsub('[$,]','',
                                          car$HOME_VAL[!is.na(car$HOME_VAL)]))
car$HOME_VAL <- as.numeric(car$HOME_VAL)
car$CLM_AMT <- as.numeric(gsub('[$,]','',car$CLM_AMT))
car$BLUEBOOK <- as.numeric(gsub('[$,]','',car$BLUEBOOK))
car$OLDCLAIM <- as.numeric(gsub('[$,]','',car$OLDCLAIM))

# Time for dealing with missing values in OCCUPATION (categorical)
# We can either:
# 1) replace the NAs with the most frequent category 
# (it can produce unreliable results)
# 2) create a new "Unknown" class for the NAs
car$OCCUPATION[is.na(car$OCCUPATION)] <- 'Unknown'
car$OCCUPATION <- as.factor(car$OCCUPATION)
levels(car$OCCUPATION)[levels(car$OCCUPATION)=="z_Blue Collar"] <- "Blue Collar"
table(car$OCCUPATION,ifelse(car$CLAIM_FLAG==1,"Claimed","Not claimed"))
car$MSTATUS <- as.factor(ifelse(gsub('z_','',car$MSTATUS) == "Yes", 1, 0))
car$PARENT1 <- as.factor(ifelse(gsub('z_','',car$PARENT1) == "Yes", 1, 0))
car$GENDER <- as.factor(ifelse(gsub('z_','',car$GENDER) == "F", 1, 0))
car$RED_CAR <- as.factor(ifelse(car$RED_CAR == "yes", 1, 0))
car$REVOKED <- as.factor(ifelse(car$REVOKED == "Yes", 1, 0))
car$CAR_USE <- as.factor(car$CAR_USE)
car$CAR_TYPE[car$CAR_TYPE == "z_SUV"] <- "SUV"
car$CAR_TYPE <- as.factor(car$CAR_TYPE)
car$CLAIM_FLAG <- as.factor(car$CLAIM_FLAG)
car$EDUCATION[car$EDUCATION == "<High School"] <- "School"
car$EDUCATION[car$EDUCATION == "z_High School"] <- "High School"
car$EDUCATION <- factor(car$EDUCATION,
                             levels = c("School",
                                        "High School",
                                        "Bachelors",
                                        "Masters",
                                        "PhD"))

# Dealing with outliers
colnames(car)
boxplot(car$INCOME)
# Let's inspect those with more than $300000 income
# All of them are male, with PHD, doctor, manager and unknown 
# without children and at least 40 y.o. So they are perfectly sensible values.
# They are extreme observation rather than outliers
subset(car,car$INCOME > 300000)

boxplot(car$HOME_VAL)
# Let's inspect those with more than $700000 home value
# Just like above, it all make sense. So no outliers here as well.
subset(car,car$HOME_VAL > 700000)

boxplot(car$BLUEBOOK)
# Let's inspect those with more than $55000 price value of their 
# car according to KBB system (Kelley Blue Book)
# Still no outliers. 
subset(car,car$BLUEBOOK > 55000)

boxplot(car$OLDCLAIM)
boxplot(car$CLM_AMT)
# The observations are so close to each other we can hardly assign any of them
# as a potential outlier. Similar situation in claim amount.


# Dividing into train/test set
set.seed(101)
df_which_train <- createDataPartition(car$CLAIM_FLAG,
                                      p = 0.7,
                                      list = FALSE)
df_train <- car[df_which_train,]
df_test <- car[-df_which_train,]

# Sorting missing values
colSums(is.na(df_train)) %>% sort()

# Visualizing the patterns for missing values
aggr(df_train, col=mdc(1:2),numbers=FALSE, sortVars=TRUE,
     labels=names(df_train), cex.axis=.46, gap=0.1, 
     ylab=c("Proportion of missingness","Missingness Pattern"))

# Drawing margin plot for some missing columns. 
# The red plot indicates distribution of one feature when 
# it is missing while the blue box is the distribution of all 
# others when the feature is present.
# TAB
marginplot(df_train[, c("INCOME","HOME_VAL")], col = mdc(1:2), 
           cex.numbers = 1.2, pch = 19)
marginplot(df_train[, c("INCOME","CAR_AGE")], col = mdc(1:2), 
           cex.numbers = 1.2, pch = 19)

#Imputing missing values for AGE, YOJ, CAR_AGE, HOME_VAL and INCOME using mice 
# based on Predictive mean matching (PMM)
imp <- mice(data = df_train[,c(2,4,5,7,23)], 
            seed = 500, 
            m = 2, # Number of multiple imputations
            method = "pmm",
            maxit = 10,
            print = FALSE)

# We can view the imputed values for missing columns for two datasets
# Two red density curves are imputed ones and the blue is the observed data
densityplot(imp)

# All except for age look promising. 
# For age it didn't capture the true underlying distribution. 
# So we impute age manually with the most common observation (45)
mode_ <- function(x){which.max(tabulate(x))}
df_train$AGE[is.na(df_train$AGE)] <- mode_(df_train$AGE)
# Using the same value from the training set for test set
df_test$AGE[is.na(df_test$AGE)] <- mode_(df_train$AGE) 

# We can impute the other four columns with the mice results
df_train$INCOME <- complete(imp,2)$INCOME
df_train$HOME_VAL <- complete(imp,2)$HOME_VAL
df_train$CAR_AGE <- complete(imp,2)$CAR_AGE
df_train$YOJ <- complete(imp,2)$YOJ

# Since we can't impute the missing values in test set 
# directly (will cause information leakage), therefore we reuse the 
# imputed values from training set for the test set with mice.reuse function
# mice.reuse function is from 
# https://github.com/prockenschaub/Misc/tree/master/R/mice.reuse
source("mice.reuse.R")
df_test_imp <- mice.reuse(imp, df_test[,c(2,4,5,7,23)], maxit = 1)
# Warnings are about different length of the training and test sets

df_test$INCOME <- df_test_imp$`2`$INCOME
df_test$HOME_VAL <- df_test_imp$`2`$HOME_VAL
df_test$CAR_AGE <- df_test_imp$`2`$CAR_AGE
df_test$YOJ <- df_test_imp$`2`$YOJ


######
# Feature selection

options(repr.plot.width = 12, repr.plot.height = 8)
plot_grid(ggplot(df_train, aes(x=GENDER,fill=CLAIM_FLAG))+ 
            geom_bar()+ theme_gray(), 
          ggplot(df_train, aes(x=MSTATUS,fill=CLAIM_FLAG))+ 
            geom_bar(position = 'fill')+theme_gray(),
          ggplot(df_train, aes(x=PARENT1,fill=CLAIM_FLAG))+ 
            geom_bar(position = 'fill')+theme_gray(),
          ggplot(df_train, aes(x=RED_CAR,fill=CLAIM_FLAG))+ 
            geom_bar(position = 'fill')+theme_gray(),
          ggplot(df_train, aes(x=REVOKED,fill=CLAIM_FLAG))+ 
            geom_bar(position = 'fill')+theme_gray(),
          ggplot(df_train, aes(x=CAR_USE,fill=CLAIM_FLAG))+ 
            geom_bar(position = 'fill')+theme_gray(),
          ggplot(df_train, aes(x=CAR_TYPE,fill=CLAIM_FLAG))+ 
            geom_bar(position = 'fill')+theme_gray(),
          ggplot(df_train, aes(x=OCCUPATION,fill=CLAIM_FLAG))+ 
            geom_bar(position = 'fill')+theme_gray(),
          ggplot(df_train, aes(x=EDUCATION,fill=CLAIM_FLAG))+ 
            geom_bar(position = 'fill')+theme_bw()+
            scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
          align = "h")

# H0: GENDER and CLAIM_FLAG are independent (no association between them)
stats::fisher.test(df_train$GENDER,df_train$CLAIM_FLAG) # Reject the null

# H0: RED_CAR and CLAIM_FLAG are independent (no association between them)
stats::fisher.test(df_train$RED_CAR,df_train$CLAIM_FLAG) # Failed to reject
# From the plot and p-value, RED_CAR would be better dropped as it has 
# no effect on predicting the target


options(repr.plot.width =12, repr.plot.height = 8, scipen=5)
plot_grid(ggplot(df_train, aes(y= HOME_VAL, x = "", fill = CLAIM_FLAG)) + 
            geom_boxplot()+ theme_bw()+ xlab(" "),
          ggplot(df_train, aes(y= INCOME, x = "", fill = CLAIM_FLAG)) + 
            geom_boxplot()+ theme_bw()+ xlab(" "),
          ggplot(df_train, aes(y= CAR_AGE, x = "", fill = CLAIM_FLAG)) + 
            geom_boxplot()+ theme_bw()+ xlab(" "),
          ggplot(df_train, aes(y= HOMEKIDS, x = "", fill = CLAIM_FLAG)) + 
            geom_boxplot()+ theme_bw()+ xlab(" "),
          ggplot(df_train, aes(y= CLM_AMT, x = "", fill = CLAIM_FLAG)) + 
            geom_boxplot()+ theme_bw()+ xlab(" "),
          ggplot(df_train, aes(y= CLM_FREQ, x = "", fill = CLAIM_FLAG)) + 
            geom_boxplot()+ theme_bw()+ xlab(" "),
          ggplot(df_train, aes(y= YOJ, x = "", fill = CLAIM_FLAG)) + 
            geom_boxplot()+ theme_bw()+ xlab(" "),
          ggplot(df_train, aes(y= MVR_PTS, x = "", fill = CLAIM_FLAG)) + 
            geom_boxplot()+ theme_bw()+ xlab(" "),
          ggplot(df_train, aes(y= TIF, x = "", fill = CLAIM_FLAG)) + 
            geom_boxplot()+ theme_bw()+ xlab(" "),
          ggplot(df_train, aes(y= TRAVTIME, x = "", fill = CLAIM_FLAG)) + 
            geom_boxplot()+ theme_bw()+ xlab(" "),
          ggplot(df_train, aes(y= OLDCLAIM, x = "", fill = CLAIM_FLAG)) + 
            geom_boxplot()+ theme_bw()+ xlab(" "),
          ggplot(df_train, aes(y= BLUEBOOK, x = "", fill = CLAIM_FLAG)) + 
            geom_boxplot()+ theme_bw()+ xlab(" "))

# The median income for individuals who claimed is around 50000
# Less aged car is more likely to be claimed
# The median home value for individuals who claimed is below 125000

# Low point-biserial correlation for YOJ, TRAVTIME and TIF
cor.test(df_train$YOJ,ifelse(as.numeric(df_train$CLAIM_FLAG) == 2,1,0))$estimate
cor.test(df_train$TRAVTIME,ifelse(as.numeric(df_train$CLAIM_FLAG) == 2,1,0))$estimate
cor.test(df_train$TIF,ifelse(as.numeric(df_train$CLAIM_FLAG) == 2,1,0))$estimate

# Pairwise correlation for numeric variables
df_new <- NULL
df_new$CLAIM_FLAG <- ifelse(as.numeric(df_train$CLAIM_FLAG) == 2, 1, 0)
df2 <- df_train[,-24]
df2$CLAIM_FLAG <- df_new$CLAIM_FLAG
df_numeric_vars <- sapply(df2, is.numeric) %>% which() %>% names()
df_correlations <- cor(df2[,df_numeric_vars],
      use = "pairwise.complete.obs")
corrplot::corrplot(df_correlations, method = 'pie')

# Let's check for high correlated columns if any
findCorrelation(df_correlations,
                cutoff = 0.75,
                names = TRUE)

# Better to drop TIF, YOJ and TRAVTIME due to their low predictive power
abs(df_correlations[,"CLAIM_FLAG"]) %>% 
  sort(decreasing = TRUE) %>%
  names()

# Visualizing high-correlated factor variables
is.fact <- sapply(df_train, is.factor)
factors.df <- df_train[, is.fact]
model.matrix(~0+., data=factors.df) %>% 
  cor(use="pairwise.complete.obs") %>% 
  ggcorrplot(show.diag = F, type="lower", lab=TRUE, lab_size=2.5)

# No near zeor variance
nearZeroVar(df_train)

# No linear combo
findLinearCombos(df_train[, df_numeric_vars[-15]] )

par(mfrow = c(1, 2))
hist(df_train$CLM_AMT, main = "Claim amount highly skewed") # Highly skewed
hist(log(df_train$CLM_AMT+1),main = "Log of claim amount") # A bit better now
dev.off()

df_train$CLM_AMT <- log(df_train$CLM_AMT+1)
df_test$CLM_AMT <- log(df_test$CLM_AMT+1)

cor(ifelse(as.numeric(df_train$CLAIM_FLAG) == 2, 1, 0),df_train$CLM_AMT)
# CLM_AMT (Claim amount) is highly correlated with the target. Even if 
# we run the regression on CLM_AMT we would obtain 100% accuracy which 
# perfectly separates 0 from 1 in target variable. Also the glm() function 
# will raise warning:glm.fit: fitted probabilities numerically 0 or 1 occurred. 
# This is called "Quasi-complete separation" and there are some solutions to it. 
# Sometimes the problem with removing predictors is that we're removing 
# the predictor that best explains the response, which is usually 
# what we aiming to do but not in this case.
# For at least not getting the aforementioned warning, We will use 
# Bayesian analysis with non-informative prior assumptions as 
# proposed by Gelman et al (2008) using the bayesglm() in arm package
table(ifelse(df_train$CLAIM_FLAG==1,"Yes","No"),
      ifelse(predict(bayesglm(CLAIM_FLAG~CLM_AMT, data=df_train, 
                              family=binomial(link='logit'),prior.df=5),
                     type = "response") > 0.5, # condition
             "Yes", # what returned if condition TRUE
             "No"))
# One can see, the CLM_AMT alone can result in a 100% accuracy 


# Logistic regression
set.seed(101)
car_logit1 <- glm(CLAIM_FLAG ~ ., # Providing all predictors at once
                  family =  binomial(link = "logit"),
                  data = df_train %>% dplyr::select(-c(CLM_AMT)))
summary(car_logit1) # AIC: 7148.9

# AGE,YOJ,GENDER, RED_CAR and CAR_AGE are insignificant at 5%
set.seed(101)
car_logit2 <- glm(CLAIM_FLAG ~ .,
                  family =  binomial(link = "logit"),
                  data = df_train %>% 
                    dplyr::select(-c(AGE,RED_CAR,CAR_AGE,CLM_AMT,YOJ,GENDER)))
summary(car_logit2) # AIC: 7142.4

# Using a binned residual plot for Logistic regression from arm package
binnedplot(fitted(car_logit2), 
           residuals(car_logit2, type = "response"), 
           nclass = NULL, 
           xlab = "Expected Values", 
           ylab = "Average residual", 
           main = "Binned residual plot", 
           cex.pts = 0.8, 
           col.pts = 1, 
           col.int = "gray")
floor(sqrt(length(fitted(car_logit2)))) # Number of binned result (nclass)
# The gray lines indicate +/- 2 standard error bounds, 
# within which one would expect about 95% of the binned residuals to fall, 
# if the model were actually true.
# As one can see there are at least 5 outlier bins (1 - 5/84 = 0.94) 
# some of them are negative.This means that in these bins the model 
# is predicting a higher average rate of CLAIM_FLAG than is actually 
# the case and vice versa for positives. 
# Since the residuals are observed minus fitted values, so if, 
# the rate of observed CLAIM_FLAG is less than what the model predicts, 
# then the residuals will be negative. So the model is over-predicting 
# CLAIM_FLAG and under-predicting for positive average residuals.
# We can take a look at the individual predictors against the binned 
# residuals to see where we might refine the model
# Building another model but this time let's transform some 
# variables using Yeo Johnson transformation which is a similar
# to Box-Cox but can also handle non-positive value (in our case 0)
yeo_johnson <- function(x, lambda) {
  eps <- 0.001
  if (abs(lambda) < eps)
    log(x + 1)
  else
    ((x + 1) ^ lambda - 1) / lambda}

lambda_CLM_FREQ <- yeojohnson(df_train$CLM_FREQ)$lambda
df_train$CLM_FREQ <- yeo_johnson(df_train$CLM_FREQ,lambda_CLM_FREQ)
# And the same needs to be apply for test set
df_test$CLM_FREQ <- yeo_johnson(df_test$CLM_FREQ,lambda_CLM_FREQ)

set.seed(101)
car_logit3 <- glm(CLAIM_FLAG ~ .,
                  family =  binomial(link = "logit"),
                  data = df_train %>% 
                    dplyr::select(-c(CLM_AMT,CAR_AGE,RED_CAR,GENDER,YOJ,AGE)))
summary(car_logit3) # AIC: 7094.1
# We can also change the link to "probit". However the results 
# will be very similar to one we had in "logit" model
# Checking the binned plot again
binnedplot(fitted(car_logit3), 
           residuals(car_logit3, type = "response"), 
           nclass = NULL, 
           xlab = "Expected Values", 
           ylab = "Average residual", 
           main = "Binned residual plot", 
           cex.pts = 0.8, 
           col.pts = 1, 
           col.int = "gray")
# Now there are at most 2 bins out of +/- 2 SE. 1- (2/84) = 0.9761
# At least 95% of binned residuals (exactly 97%) falling in the +/- 2 SE range

# Evaluation so far
fitted <- predict(car_logit3,type = "response")
table(ifelse(df_train$CLAIM_FLAG == 1,"Yes","No"),
      ifelse(fitted > 0.5,"Yes","No"))
clog2 <- as.factor(ifelse(fitted >0.5,"Yes","No"))
confusionMatrix(clog2, as.factor(ifelse(df_train$CLAIM_FLAG == 0,"No","Yes")),
                positive = "Yes")

# If we want to increase the sensitivity we can change the cutoff threshold 
# at the cost of lowering the accuracy and specificity.
# We can plot the curve for accuracy, sensitivity and specificity, then
# locate the point that these three metrics intersect with each other 

# A function for evaluating metrics for different cutoff point
perform_fn <- function(cutoff) {
  predicted_claim <- factor(ifelse(fitted >= cutoff, "Yes", "No"))
  actual_claim <- factor(ifelse(df_train$CLAIM_FLAG==1,"Yes","No"))
  conf <- confusionMatrix(predicted_claim, actual_claim, positive = "Yes")
  accuray <- conf$overall[1]
  sensitivity <- conf$byClass[1]
  specificity <- conf$byClass[2]
  out <- t(as.matrix(c(sensitivity, specificity, accuray))) 
  colnames(out) <- c("sensitivity", "specificity", "accuracy")
  return(out)}

# Preparing the plot and creating a matrix for storing three metric values
options(repr.plot.width =8, repr.plot.height =6)
s = seq(0.01,0.80,length=100) # Hundred cutoff points from 0.01 to 0.80
OUT = matrix(0,100,3)
for(i in 1:100){
  OUT[i,] = perform_fn(s[i])
} 
# Building the plot sequentially 
plot(s, OUT[,1],xlab="Cutoff",ylab="Value",cex.lab=1.5,cex.axis=1.5,
     ylim=c(0,1),type="l",lwd=2,axes=FALSE,col=2, main = "Metrics by cutoff")
axis(1,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
axis(2,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
lines(s,OUT[,2],col="darkgreen",lwd=2)
lines(s,OUT[,3],col=4,lwd=2)
box()
legend("bottom",col=c(2,"darkgreen",4,"darkred"),text.font =3,inset = 0.02,
       box.lty=0,cex = 0.8, 
       lwd=c(2,2,2,2),c("Sensitivity","Specificity","Accuracy"))
abline(v = 0.265, col="red", lwd=1, lty=2)

# Now let's use the new cutoff point (0.265)
table(ifelse(df_train$CLAIM_FLAG == 1,"Yes","No"),
      ifelse(fitted > 0.265,"Yes","No"))
clog2 <- as.factor(ifelse(fitted >0.265,"Yes","No"))
confusionMatrix(clog2, as.factor(ifelse(df_train$CLAIM_FLAG == 0,"No","Yes")),
                positive = "Yes")

# The sensitivity increased to around 70%, accuracy also dropped to around 70%
# Let's get back to the good ol' 0.5 as it's the default value most of the time

# representing the measures
summary_binary <- function(predicted_probs,
                           real,
                           cutoff = 0.5,
                           level_positive = "Yes",
                           level_negative = "No") {
  ctable <- confusionMatrix(as.factor(ifelse(fitted >cutoff, 
                                             level_positive, 
                                             level_negative)), 
                            as.factor(ifelse(real == 1,"Yes","No")), 
                            level_positive) 
  stats <- round(c(ctable$overall[1],
                   ctable$byClass[c(1:4, 7, 11)]),
                 2)
  return(stats)
}

summary_binary(predicted_probs = fitted,
               real = df_train$CLAIM_FLAG)

# ROC curve and area
roc.plot((ifelse(as.numeric(df_train$CLAIM_FLAG) == 2,1,0)),
         fitted,threshold = seq(0.1,0.9, 0.1)) 
roc.area((ifelse(as.numeric(df_train$CLAIM_FLAG) == 2,1,0)),
         fitted)$A 

# Cross validation
ctrl_nocv <- trainControl(method = "none")
set.seed(101)
car_logit2_train <- 
  train(CLAIM_FLAG ~ .,
        data = df_train %>% 
          dplyr::select(-c(AGE,RED_CAR,CAR_AGE,CLM_AMT,YOJ,GENDER)),
        method = "glm",
        family = "binomial",
        trControl = ctrl_nocv)

car_logit2_fitted <- predict(car_logit2_train,
                             df_train,
                             type = "prob")
# Accuracy = 77% on train data
confusionMatrix(as.factor(ifelse(car_logit2_fitted$`1` > 0.5 ,"Yes","No")), 
                as.factor(ifelse(df_train$CLAIM_FLAG == 0,"No","Yes")),
                positive = "Yes")

# lets see the forecast error on the TEST sample
car_logit2_forecasts <- predict(car_logit2_train,
                                df_test,
                                type = "prob")
# Accuracy around 78% on test data
confusionMatrix(as.factor(ifelse(car_logit2_forecasts$`1` > 0.5 ,"Yes","No")), 
                as.factor(ifelse(df_test$CLAIM_FLAG == 0,"No","Yes")),
                positive = "Yes")
# ROC/AUC on test data close to 64%
roc.area((ifelse(as.numeric(df_test$CLAIM_FLAG) == 2,1,0)),
         ifelse(car_logit2_forecasts$`1` > 0.5 ,1,0))$A

# CV
ctrl_cv10 <- trainControl(method = "cv",number = 10, 
                          classProbs = TRUE,summaryFunction = twoClassSummary)

# For easier refrence
df_train$CLAIM_FLAG <- as.factor(ifelse(df_train$CLAIM_FLAG == 1,"Yes","No"))
df_test$CLAIM_FLAG <- as.factor(ifelse(df_test$CLAIM_FLAG == 1,"Yes","No"))

set.seed(101)
car_logit2_train2 <- 
  train(CLAIM_FLAG ~ .,
        data = df_train %>% 
          dplyr::select(-c(AGE,RED_CAR,CAR_AGE,CLM_AMT,YOJ,GENDER)),
        method = "glm",
        family = "binomial",
        metric = "ROC",
        trControl = ctrl_cv10)
car_logit2_train2

# Maybe LOOCV? NO, it takes too much time!
# ctrl_loocv <- trainControl(method = "LOOCV", classProbs = TRUE,
# summaryFunction = twoClassSummary)
# 
# set.seed(101)
# car_logit2_train3 <- 
#   train(CLAIM_FLAG ~ .,
#         data = df_train %>% 
# dplyr::select(-c(AGE,RED_CAR,CAR_AGE,CLM_AMT,YOJ,GENDER)),
#         method = "glm",
#         family = "binomial",
#         metric = "ROC",
#         trControl = ctrl_loocv)
# ROC        Sens       Spec     
# 0.7550005  0.9365229  0.2947977
# car_logit2_train3 

# Now with repeated CV
ctrl_cv5x3a <- trainControl(method = "repeatedcv",number = 5,classProbs = TRUE,
                            summaryFunction = twoClassSummary,repeats = 3)
set.seed(101)
car_logit2_train3 <- 
  train(CLAIM_FLAG ~ ., 
        data = df_train %>% 
          dplyr::select(-c(AGE,RED_CAR,CAR_AGE,CLM_AMT,YOJ,GENDER)),
        method = "glm",
        family = "binomial",
        metric = "ROC",
        trControl = ctrl_cv5x3a)
car_logit2_train3 # Not much of a difference

# TRUE
identical(coef(car_logit2_train$finalModel),
          coef(car_logit2_train3$finalModel))
identical(coef(car_logit2_train2$finalModel),
          coef(car_logit2_train3$finalModel))

# Standard residuals
# Data points with an absolute standardized residuals 
# above 3 represent possible outliers which in our case there's none
model.data <- augment(car_logit3) %>% 
  mutate(index = 1:n())
# Deviance residual
plot(residuals.glm(car_logit3,type = "deviance"), ylab= "Deviance residual")
# A nicer representation based on ggplot
ggplot(model.data, aes(index, .std.resid)) + 
  geom_point(aes(color = CLAIM_FLAG), alpha = .5) +
  theme_bw()
# 0 outliers 
model.data %>% 
  filter(abs(.std.resid) > 3)


# KNN
# First let's use k=5 (a random number)
set.seed(101)
car_train_knn5 <- 
  # From the feature selection we identified the less effective variables
  train(CLAIM_FLAG ~ .-CLM_AMT -TIF -TRAVTIME -RED_CAR -YOJ, 
        data = df_train,
        method = "knn",
        tuneGrid = data.frame(k = 5))

car_train_knn5_fitted <- predict(car_train_knn5,
                                 df_train)
source("F_summary_binary_class.R")
summary_binary_class(predicted_classes = car_train_knn5_fitted,
                     real = df_train$CLAIM_FLAG)

# Nearest odd number to avoid ties would be k=85
k_value <- data.frame(k = round(sqrt(dim(df_train)[1])))

# 5 fold repeated CV and k=85
ctrl_cv5x3a <- trainControl(method = "repeatedcv",number = 5,repeats = 3)
set.seed(101)
car_train_knn85 <- 
  train(CLAIM_FLAG ~ .-CLM_AMT -TIF -TRAVTIME -RED_CAR -YOJ, 
        data = df_train,
        method = "knn",
        trControl = ctrl_cv5x3a,
        tuneGrid = k_value)

# Fitted values - prediction on the training sample
car_train_knn85_fitted <- predict(car_train_knn85,
                                  df_train)

# Almost everything decreased!
summary_binary_class(predicted_classes = car_train_knn85_fitted,
                     real = df_train$CLAIM_FLAG) 

# Comparing the results in the test sample
# About the same result as we had in training set
car_train_knn85_forecasts <- predict(car_train_knn85,
                                     df_test)
summary_binary_class(predicted_classes = car_train_knn85_forecasts,
                     real = df_test$CLAIM_FLAG)

# Now we experiment with different values for k
different_k <- data.frame(k = seq(5, 97, 4))

# Here comes the pain
ctrl_cv3x2a <- trainControl(method = "repeatedcv",number = 3,classProbs = TRUE,
                            summaryFunction = twoClassSummary,repeats = 2)
set.seed(101)
car_train_knn_tuned <- 
  train(CLAIM_FLAG ~ .-CLM_AMT -TIF -TRAVTIME -RED_CAR -YOJ, 
        data = df_train,
        method = "knn",
        metric = "ROC",
        trControl = ctrl_cv3x2a,
        tuneGrid = different_k)

# Plotting the values for k
plot(car_train_knn_tuned) # k = 93
# CV
ctrl_cv5 <- trainControl(method = "cv",number = 5, 
                         classProbs = TRUE,summaryFunction = twoClassSummary)

# Optimal k based on scaled variables from range [0,1] since
# rescaling features is one way that can be used to improve the performance 
# of Distance-based algorithms such as KNN
set.seed(101)
car_train_knn_tuned_scaled <- 
  train(CLAIM_FLAG ~ .-CLM_AMT -TIF -TRAVTIME -RED_CAR -YOJ, 
        data = df_train,
        method = "knn",
        trControl = ctrl_cv5,
        tuneGrid = different_k,
        preProcess = c("range"),
        metric = "ROC")

# Plotting the values for k
plot(car_train_knn_tuned_scaled) # k = 89

# Storing all the prediction in a dataframe
car_test_forecasts <- 
  data.frame(car_train_knn5 = predict(car_train_knn5,
                                                   df_test),
             car_train_knn85 = predict(car_train_knn85,
                                       df_test),
             car_train_knn_tuned = predict(car_train_knn_tuned,
                                           df_test),
             car_train_knn_tuned_scaled = predict(car_train_knn_tuned_scaled,
                                                  df_test))
source("F_summary_binary.R")
sapply(car_test_forecasts,
       function(x) summary_binary_class(predicted_classes = x,
                                        real = df_test$CLAIM_FLAG)) %>%t()
# One can see the last model which we trained on CV with scaled 
# variable has the highest accuracy but suffers from low sensitivity 


# SVM

# Linear SVM
trctrl <- trainControl(method = "cv", number = 5)
# What if we include CLM_AMT? 100% accuracy.
set.seed(101)
svm_Linear <- train(CLAIM_FLAG ~.-TIF -TRAVTIME -RED_CAR -YOJ, data = df_train, 
                    method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
svm_Linear$results

set.seed(101)
svm_Linear <- train(CLAIM_FLAG ~.-CLM_AMT -TIF -TRAVTIME -RED_CAR -YOJ, 
                    data = df_train, 
                    method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
svm_Linear$results

# Not run
# grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 
# 0.75, 1, 1.25, 1.5, 1.75, 2,5))
# svm_Linear_Grid <- train(CLAIM_FLAG ~.-CLM_AMT -TIF -TRAVTIME -RED_CAR -YOJ, 
#                          data = df_train, 
#                          method = "svmLinear",
#                          trControl=trctrl,
#                          preProcess = c("center", "scale"),
#                          tuneGrid = grid,
#                          tuneLength = 10)
# svm_Linear_Grid # Optimal C = 1.75, Accuracy = 0.7542989
# Prediction on test set:
# Accuracy : 0.7547  Sensitivity : 0.9453   Specificity : 0.2467
# test_pred_grid <- predict(svm_Linear_Grid, newdata = df_test)
# confusionMatrix(table(test_pred_grid, df_test$CLAIM_FLAG))

# Polynomial SVM
# Not run
# svm_parametersPoly <- expand.grid(C = c(1.75),
#                                   degree = 2:4,
#                                   scale = 1)
# trctrl <- trainControl(method = "cv", number = 3)
# set.seed(101)
# svm_Poly <- train(CLAIM_FLAG ~.-CLM_AMT -TIF -TRAVTIME -RED_CAR -YOJ, 
#                     data = df_train,
#                     method = "svmPoly",
#                     trControl=trctrl,
#                     tuneGrid = svm_parametersPoly)
# 
# svm_Poly
# Accuracy 0.7493415
# The final values used for the model were degree = 2, scale = 1 and C = 1.75

# test_pred_grid <- predict(svm_Poly, newdata = df_test)
# confusionMatrix(table(test_pred_grid, df_test$CLAIM_FLAG))
# Accuracy : 0.7659
# Sensitivity : 0.9012
# Specificity : 0.3369

# RBF 
# First without any CV
set.seed(101)
car.svm_Radial1 <- train(CLAIM_FLAG ~.-CLM_AMT -TIF -TRAVTIME -RED_CAR -YOJ, 
                         data = df_train, 
                         method = "svmRadial",
                         trControl = ctrl_nocv)
car.svm_Radial1$finalModel@error # High training error : 0.234854
# Accuracy : 0.7572 
confusionMatrix(predict(car.svm_Radial1,df_test),
                df_test$CLAIM_FLAG, positive = "Yes")

# Now with CV and different values for C and Sigma
parametersC_sigma <-
  expand.grid(C = c(2,5),
              sigma = c(0.01,0.02))
trctrl <- trainControl(method = "cv", number = 3)

set.seed(101)
car.svm_Radial2 <- train(CLAIM_FLAG ~.-CLM_AMT -TIF -TRAVTIME -RED_CAR -YOJ,
                         data = df_train,
                         method = "svmRadial",
                         tuneGrid = parametersC_sigma,
                         trControl = trctrl)
car.svm_Radial2$finalModel@error
# The model has less training error compare to the previous one and also the 
# accuracy improved a bit. The hyperparameters for the final optimal model are:
# C = 2 and sigma =  0.02
# Accuracy : 0.7663
confusionMatrix(predict(car.svm_Radial2,df_test),
                df_test$CLAIM_FLAG, positive = "Yes")


# XGBoost

# We need to sequentially tune the hyperparameters
params_xgboost <- expand.grid(nrounds = seq(20, 100, 10),
                              max_depth = c(6),
                              eta = c(0.5), 
                              gamma = 1,
                              colsample_bytree = c(0.3),
                              min_child_weight = c(100),
                              subsample = 0.8)

ctrl_cv5x3 <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 3,
                           classProbs = TRUE)
set.seed(101)
car_xgboost_tuned <- train(CLAIM_FLAG ~.-CLM_AMT -TIF -TRAVTIME -RED_CAR -YOJ, 
                           data = df_train,
                           method = "xgbTree",
                           trControl = ctrl_cv5x3,
                           tuneGrid  = params_xgboost)
car_xgboost_tuned
plot(car_xgboost_tuned)

# One can see the value of optimal nround is low (30) so we can
# modifying the eta (to lower learning rate) and choose higher max depth
params_xgboost1 <- expand.grid(nrounds = seq(30, 100, 10),
                               max_depth = c(8),
                               eta = c(0.17), 
                               gamma = 1,
                               colsample_bytree = c(0.3),
                               min_child_weight = c(100),
                               subsample = 0.8)
set.seed(101)
car_xgboost_tuned1 <- train(CLAIM_FLAG ~.-CLM_AMT -TIF -TRAVTIME -RED_CAR -YOJ, 
                            data = df_train,
                            method = "xgbTree",
                            trControl = ctrl_cv5x3,
                            tuneGrid  = params_xgboost1)
car_xgboost_tuned1 #nrounds = 90
plot(car_xgboost_tuned1)

# Now let's tune tree parameters 
# (max_depth, min_child_weight and colsample_bytree)
params_xgboost2 <- expand.grid(nrounds = 90,
                               max_depth = seq(90, 130, 10),
                               eta = c(0.17), 
                               gamma = 1,
                               colsample_bytree = c(0.3),
                               min_child_weight = seq(50, 80, 10),
                               subsample = 0.8)
set.seed(101)
car_xgboost_tuned2 <- train(CLAIM_FLAG ~.-CLM_AMT -TIF -TRAVTIME -RED_CAR -YOJ, 
                            data = df_train,
                            method = "xgbTree",
                            trControl = ctrl_cv5x3,
                            tuneGrid  = params_xgboost2)
car_xgboost_tuned2 
plot(car_xgboost_tuned2)
# min_child_weight = 70 and max_depth = 110
# Now tuning the colsample_bytree

params_xgboost3 <- expand.grid(nrounds = 70,
                               max_depth = 110,
                               eta = c(0.17), 
                               gamma = 1,
                               colsample_bytree = seq(0.3, 0.9, 0.1),
                               min_child_weight = 70,
                               subsample = 0.8)
set.seed(101)
car_xgboost_tuned3 <- train(CLAIM_FLAG ~.-CLM_AMT -TIF -TRAVTIME -RED_CAR -YOJ, 
                            data = df_train,
                            method = "xgbTree",
                            trControl = ctrl_cv5x3,
                            tuneGrid  = params_xgboost3)
car_xgboost_tuned3 # colsample_bytree = 0.8
plot(car_xgboost_tuned3)

# Checking for optimal sample size
params_xgboost4 <- expand.grid(nrounds = 70,
                               max_depth = 110,
                               eta = c(0.17), 
                               gamma = 1,
                               colsample_bytree = 0.8,
                               min_child_weight = 70,
                               subsample = c(0.6, 0.7, 0.75,0.8,0.85,0.9,0.95))
set.seed(101)
car_xgboost_tuned4 <- train(CLAIM_FLAG ~.-CLM_AMT -TIF -TRAVTIME -RED_CAR -YOJ, 
                            data = df_train,
                            method = "xgbTree",
                            trControl = ctrl_cv5x3,
                            tuneGrid  = params_xgboost4)

car_xgboost_tuned4 #subsample = 0.75
plot(car_xgboost_tuned4)

# Now let's modify the nrounds and eta proportionally, 
# that is decreasing the learning rate (eta) 
# by 1/5 and multiply 5 times the number of trees (nrounds)

params_xgboost5 <- expand.grid(nrounds = 350,
                               max_depth = 110,
                               eta = c(0.035), 
                               gamma = 1,
                               colsample_bytree = 0.8,
                               min_child_weight = 70,
                               subsample = 0.75)

set.seed(101)
car_xgboost_tuned5 <- train(CLAIM_FLAG ~.-CLM_AMT -TIF -TRAVTIME -RED_CAR -YOJ, 
                            data = df_train,
                            method = "xgbTree",
                            trControl = ctrl_cv5x3,
                            tuneGrid  = params_xgboost5)
car_xgboost_tuned5
# Now let's compare the models
source("accuracy_ROC_model.R")
models <- paste0("car_xgboost_tuned", c("", "1":"5"))

sapply(models,
       function(x) accuracy_ROC_model(model = get(x),
                                      data = df_test,
                                      target_variable = "CLAIM_FLAG",
                                      predicted_class = "Yes")) %>% t()
# The last model which is the most tuned (car_xgboost_tuned5) 
# reached the higher accuracy and sensitivity compare to the others.


# Conclusion
# One can see all those 4 algorithms perform so close to each other based on
# metrics such as accuracy, specificity and sensitivity. However the most
# efficient one was XGBoost due to it's fast running time and accuracy but
# there was not much difference between XGBoost and logistic regression so
# sometimes it's better to come up with a simpler approach rather than a
# more complex one.