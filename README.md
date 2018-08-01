# DataScience

telco<-read.csv("https://raw.githubusercontent.com/vmanuelguevara/DataScience/master/Perdida%20de%20clientes.csv", sep = ";")
head(telco)

#El objetivo es predecir si un cliente va a desafiliarse de la compañia o no

summary(telco)
dim(telco)

#Dividimos el conjunto de datos en train y test
#Tomamos el 75% de la data para train
#El resto para la data de test

div= sort(sample(nrow(telco), nrow(telco)*.75))
train<-telco[div,]
test<-telco[-div,] 

dim(train)
dim(test)

#REALIZAMOS EL PRIMER MODELO APLICANDO REGRESIÓN LOGISTICA

modelo1 <- glm(Desafiliado ~ ., data = telco, family = "binomial")
summary(modelo1)

modelo2<-step(modelo1)

#Prediccion
prob<-predict(modelo1,type="response")
res<-residuals(modelo1, type = "deviance")

plot(predict(modelo1), res,
     xlab="Fitted values", ylab = "Residuals",
     ylim = max(abs(res)) * c(-1,1))

modelo1_score
#score test data set
library(ROCR)
train$modelo1_score <- predict(modelo1,type='response',train)
m1_pred <- prediction(train$modelo1_score, train$Desafiliado)
m1_perf <- performance(m1_pred,"tpr","fpr")
?performance

# Vieindo la curva ROC
plot(m1_perf, lwd=2, colorize=TRUE, main="ROC m1: Logistic Regression Performance")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

#KS, Gini & AUC m1
m1_KS <- round(max(attr(m1_perf,'y.values')[[1]]-attr(m1_perf,'x.values')[[1]])*100, 2)
m1_AUROC <- round(performance(m1_pred, measure = "auc")@y.values[[1]]*100, 2)
m1_Gini <- (2*m1_AUROC - 100)
cat("AUROC: ",m1_AUROC,"\tKS: ", m1_KS, "\tGini:", m1_Gini, "\n")


library(randomForest)
modelo3<-randomForest(Desafiliado ~.-Desafiliado, data = train)

prob2<-predict(modelo3,type="response")
m3_fitForest <-predict(modelo3,newdata=test, type="prob")[,2]


m3 <- randomForest(Desafiliado ~ .-Desafiliado, data = train)

m3_fitForest <- predict(m3, newdata = test, type="prob")[,2]
m3_pred <- prediction( m3_fitForest, test$Desafiliado)
m3_perf <- performance(m3_pred, "tpr", "fpr")

#plot variable importance
varImpPlot(m3, main="Random Forest: Variable Importance")

plot(m3_perf,colorize=TRUE, lwd=2, main = "m3 ROC: Random Forest", col = "blue")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

#KS & AUC  m3
m3_AUROC <- round(performance(m3_pred, measure = "auc")@y.values[[1]]*100, 2)
m3_KS <- round(max(attr(m3_perf,'y.values')[[1]] - attr(m3_perf,'x.values')[[1]])*100, 2)
m3_Gini <- (2*m3_AUROC - 100)
cat("AUROC: ",m3_AUROC,"\tKS: ", m3_KS, "\tGini:", m3_Gini, "\n")


library(kernlab)

m7_2 <- ksvm(Desafiliado ~ .-Desafiliado, data = train, kernel = "rbfdot")
m7_2_pred <- predict(m7_2, test, type="response")
# Model accuracy:
table(m7_2_pred, test$Desafiliado)

m7_2_score <- predict(m7_2,test, type="decision")
m7_2_pred <- prediction(m7_2_score, test$Desafiliado)

m7_2_perf <- performance(m7_2_pred, measure = "tpr", x.measure = "fpr")
plot(m7_2_perf, colorize=TRUE, lwd=2, main="SVM:Plot ROC curve - RBF", col="blue")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)


# Model Performance
#KS &AUC m7_2
m7_2_AUROC <- round(performance(m7_2_pred, measure = "auc")@y.values[[1]]*100, 2)
m7_2_KS <- round(max(attr(m7_2_perf,'y.values')[[1]]-attr(m7_2_perf,'x.values')[[1]])*100, 2)
m7_2_Gini <- (2*m7_2_AUROC - 100)
cat("AUROC: ",m7_2_AUROC,"\tKS: ", m7_2_KS, "\tGini:", m7_2_Gini, "\n")
