# Loop through columns 3 to 10
function(data){
  data<-data
  
  dataColNames <- names(data)  # Get column names of the mtcars dataset
  
  #summary report 
  summary_df <- data.frame(
    Model = character(),
    Adj_R_Squared=numeric(),
    AIC=numeric(),
    BIC=numeric(),
    SignificantPredictors = numeric())
  
  
  
  
  # model: mpg~cyl
  
  fitCyl<- lm(~cyl,data = mtcars)
  SummaryCyl<-summary(fitCyl) #all variables
  SummaryCyl$coefficients
  
  
  summary_df<-rbind(summary_df,
                    data.frame(Model= "mpg~cyl",
                               Adj_R_Squared= SummaryCyl$adj.r.squared,
                               AIC=AIC(fitCyl),
                               BIC=BIC(fitCyl),
                               SignificatPredictors=1))
  
  
  
  
  
  MTCARSdataColNames <- names(mtcars)  # Get column names of the mtcars dataset
  
for (i in 3:length(data)) {
  #single variable models
  print(paste("Single Model: ", paste0("mpg~",MTCARSdataColNames[i])))
  singleModelFit<-lm(as.formula(paste0("mpg~",MTCARSdataColNames[i])),data=mtcars)
  SummarySingle<-summary(singleModelFit)
  coefSingleModel<-SummarySingle$coefficients
  summary_df<-rbind(summary_df,data.frame(
    Model= paste0("mpg~",MTCARSdataColNames[i]),
    Adj_R_Squared=SummarySingle$adj.r.squared,
    AIC=AIC(singleModelFit),
    BIC=BIC(singleModelFit),
    SignificatPredictors=1))
  
  
  #if (coefSingleModel[2,4] <= 0.05) {}
  print("----------------------------------------------------------")
  #multiple variable models
  model <- paste0("+", paste(MTCARSdataColNames[3:i], collapse = " + "))
  model2<-paste0("mpg~ cyl",model)
  fit<-lm(as.formula(model2),data=mtcars)
  coef<-summary(fit)$coefficients
  rsqr<-summary(fit)$adj.r.squared
  #print("Model mpg~.")
  #print(rawCoef)
  report<-0
  
  for (p_value in coef[, 4]) {
    if (p_value <= 0.05) {
      report <- report + 1
    }
  }
  summary_df<-rbind(summary_df,data.frame(Model= model2,
                                          Adj_R_Squared=rsqr, 
                                          AIC=AIC(fit),
                                          BIC=BIC(fit),
                                          SignificatPredictors=report))
  
  print(paste("Number of significant predictors:", report))
  print(paste("Significant Model:", model2))
  print(paste0("Model"," ",model2))
  print(coef)
  
  print("--------------------------------------------------------")
}
}
