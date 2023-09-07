library(MetaboAnalystR)
#a file where the first 874 columns are the high confidence metabolites, and the subsequent columns are binary indicators of whether or not drugs were taken during pregnancy to perform pathway analysis
metabs <- read.csv('...')
data <- metabs[1:874]
outcomes <- colnames(metabs)[875:dim(metabs)[2]]
counter = 0
analyze <- function(outcome, metabs, data, counter){
  setwd('...')
  path = paste0(getwd(),paste0('/Desktop/EMR_Metabolite_Connection/Maternal_Features/',outcome))
  dir.create(path)
  setwd(path)
  df <- cbind(metabs[outcome],data)
  write.csv(df, './data.csv')
  mSet<-InitDataObjects("conc", "msetqea", FALSE)
  mSet<-Read.TextData(mSet, "./data.csv", "rowu", "disc");
  mSet<-SanityCheckData(mSet)
  mSet<-ReplaceMin(mSet);
  mSet<-CrossReferencing(mSet, "hmdb");
  mSet<-CreateMappingResultTable(mSet)
  mSet<-PreparePrenormData(mSet)
  mSet<-Normalization(mSet, "SumNorm", "LogNorm", "AutoNorm", ratio=FALSE, ratioNum=20)
  mSet<-PlotNormSummary(mSet, "norm_0_", "png", 72, width=NA)
  mSet<-PlotSampleNormSummary(mSet, "snorm_0_", "png", 72, width=NA)
  mSet<-SetMetabolomeFilter(mSet, F);
  mSet<-SetCurrentMsetLib(mSet, "kegg_pathway", 2);
  mSet<-CalculateGlobalTestScore(mSet)
  mSet<-PlotQEA.Overview(mSet, "qea_0_", "net", "png", 72, width=NA)
  mSet<-PlotEnrichDotPlot(mSet, "qea", "qea_dot_0_", "png", 72, width=NA)
}

for(outcome in outcomes) {
  counter = counter + 1
  analyze(outcome, metabs, data, counter)
}
