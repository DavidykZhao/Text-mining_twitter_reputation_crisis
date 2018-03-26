Needed <- c("tm", "SnowballCC", "RColorBrewer", "ggplot2", "wordcloud", "biclust", 
            "cluster", "igraph", "fpc")
install.packages(Needed, dependencies = TRUE)
install.packages("arules")
install.packages("Rcampdf", repos = "http://datacube.wu.ac.at/", type = "source")
install.packages('SnowballC')
install.packages("xlsx")
install.packages("Amelia")
install.packages("scales")
install.packages("caret")
install.packages("tidytext")
installed.packages("stringr")
install.packages("graph")
source("http://bioconductor.org/biocLite.R")
biocLite("Rgraphviz")

library(scales)
library(tm)
library("nnet")
library(tidyverse)
library(SnowballC)
library(lubridate)
library(qdapRegex)
library(wordcloud)
library(cluster)
library(arules)
library(ggplot2)
library(xlsx)
library(e1071)
library(caret)
library(tidytext)
library(stringr)
library(graph)
library("Rgraphviz")


setwd("/Users/zhaoyikai/Desktop/R_project/twitter_datasource")


##### import and combine all the sources from all dates
#####
twitter_all <-
  list.files(pattern="0*.csv") %>% 
  map_df(~read_csv2(.))
twitter_all = as.tibble(twitter_all)
## create timestamp & visualize the time distribution
twitter_all <- twitter_all %>% 
  mutate(timestamp = ymd_hms(date))
twitter_all <- twitter_all %>% 
  mutate(year = year(date))
## order twitter_all according to timestamp
twitter_all <- twitter_all[order(twitter_all$timestamp),]
## trim the twitter_all data 
twitter_all <- twitter_all %>% 
  filter(year >= 2017)
#####
#####

##How many are in the 28th
twentyeight <-  twitter_all %>%
  filter(day(timestamp) == 28)
  


twitter_sample <- twitter_all[100*(1:2596),]
twitter_sample <- twitter_sample[order(twitter_sample$timestamp),]
twitter_sample <- as.tibble(twitter_sample)
twitter_sample_300 <- twitter_sample[sample(nrow(twitter_sample), 300),]
write.table(twitter_sample_300, file="sentiment_traintest.csv",sep=",",row.names=F)
sentiment_train_test <- read_csv("sentiment_traintest.csv")

twitter_earlydates <- read.csv2("0821-0822.csv")
twitter_earlydates <- twitter_earlydates[30:(1:13), ]
write.table(twitter_earlydates, file="additive_sentiment_traintest.csv",sep=",",row.names=F)

twitter_earlydates <- read.csv2("0821-0822.csv")
twitter_earlydates <- twitter_earlydates[30:(2:14), ]
write.table(twitter_earlydates, file="additive_sentiment_traintest2.csv",sep=",",row.names=F)

## filter out the sentiments that are not NA

sentiment_train_test <-  sentiment_train_test %>% 
   drop_na(sentiment)

## tokenlization and machine learning
corpus_ml <- Corpus(VectorSource(sentiment_train_test$text))
corpus_ml_clean <- corpus_ml %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeWords, c("https.*$",
                              "pic.twitter.*$","pictwitter.*$",
                              "will","joel","joelosteen")) %>%
  tm_map(content_transformer(rm_twitter_url))

dtm_ml <- DocumentTermMatrix(corpus_ml_clean)

inspect(dtm_ml[1:10,11:20])
dim(dtm_ml)

test_index <- sample(1:nrow(sentiment_train_test), 80) 

sentiment_train <- sentiment_train_test[-test_index,]
sentiment_test <- sentiment_train_test[test_index,]
dtm_ml_test <- dtm_ml[test_index, ]
dtm_ml_train <- dtm_ml[-test_index,]
corpus_ml_clean_train <- corpus_ml_clean[-test_index]
corpus_ml_clean_test <- corpus_ml_clean[test_index]


fivefreq <- findFreqTerms(dtm_ml_train,2)
length((fivefreq))

dtm_train_controlled <- DocumentTermMatrix(corpus_ml_clean_train, control=list(dictionary = fivefreq))
dim(dtm_train_controlled)
dtm_test_controlled <- DocumentTermMatrix(corpus_ml_clean_test, control=list(dictionary = fivefreq))

convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels= c(0,1), labels=c("No", "Yes"))
  y
}

train_NB <- apply(dtm_train_controlled, 2, convert_count)
test_NB <- apply(dtm_test_controlled, 2, convert_count)

## train the model 
classifier <- naiveBayes(train_NB, 
            as.factor(sentiment_train$sentiment), laplace = 1)
class <- as.factor(sentiment_train$sentiment)
pred <- predict(classifier, newdata = test_NB)

conf.mat <- confusionMatrix(pred, sentiment_test$sentiment)
conf.mat



## plot the whole sentiment on all of the datasets
corpus_twitter_all <- Corpus(VectorSource(twitter_all$text))
corpus_all_clean <- corpus_twitter_all %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeWords, c("https.*$",
                        "pic.twitter.*$","pictwitter.*$",
                        "will","joel","joelosteen")) 
dtm_twitter_all <- DocumentTermMatrix(corpus_all_clean, control=list(dictionary = fivefreq))
twitter_all_NB <- apply(dtm_twitter_all, 2, convert_count)

pred_all <- predict(classifier, newdata = twitter_all_NB)
twitter_all <- twitter_all %>%
  mutate(sentiment = pred_all)


posnegtime <- twitter_all %>% 
  group_by(timestamp = cut(timestamp, breaks="12 hour")) %>%
  count(sentiment)

neg_pos_plot <- posnegtime %>% 
  ggplot(aes(x = timestamp, y = n,  group = sentiment))+
  geom_line(size = 1,alpha = 0.7, aes(color = sentiment)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


timestamp_plot <- ggplot(twitter_all, aes(x = timestamp)) +
  geom_histogram(position = "identity", bins = 200, show.legend = FALSE)+
  scale_x_datetime(breaks = date_breaks("12 hour"), 
                   minor_breaks=date_breaks("6 hour"),
                   labels=date_format("%d-%H"))+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))



plot <- ggplot()+
  geom_line(data = posnegtime,size = 1,alpha = 0.7, 
            aes(x = timestamp, y = n,  group = sentiment, color = sentiment)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  geom_histogram(data = twitter_all, 
                 position = "identity", bins = 200, show.legend = FALSE) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

## export according to dates
write.table(twitter, file="twitter_test.csv",sep=",",row.names=F)





## visualize twitter_all
timestamp_plot <- ggplot(twitter_all, aes(x = timestamp)) +
  geom_histogram(position = "identity", bins = 200, show.legend = FALSE)+
scale_x_datetime(breaks = date_breaks("12 hour"), 
                 minor_breaks=date_breaks("6 hour"),
                 labels=date_format("%d-%H"))+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

## find the most share/likes
## write a functin finding them
top_influence <- function(d) {
  twitter_all %>% 
    filter(day(date) == d) %>%
    mutate(influence = favorites + 2 * retweets) %>%
    arrange(influence) %>%
    select(username, date, retweets, favorites, text, influence,timestamp) %>%
    unique() %>%
    top_n(50,influence) %>%
    select(username,influence,text,timestamp)
}

top_influence_tweets <- rbind(top_influence(21),
                     top_influence(22),
                     top_influence(23),
                     top_influence(24),
                     top_influence(25),
                     top_influence(26),
                     top_influence(27),
                     top_influence(28),
                     top_influence(29),
                     top_influence(30),
                     top_influence(31),
                     top_influence(01),
                     top_influence(02),
                     top_influence(03),
                     top_influence(04)
)

top_20 <- top_n(top_influence_tweets, 50, wt = influence)

write.table(top_20, file="top_20_influence.csv",sep=",",row.names=F)
## plot of 50 most influencial tweets 
ggplot(top_20, aes(timestamp, influence,fill = username)) +
  geom_jitter(size = 3, shape = 21, alpha = 0.5) +
  theme_bw(10) + xlab("timestamp") + ylab("influence") +
  ggtitle("top50 influence tweets")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  scale_y_continuous(limits=c(0, 130000))

## plot the influece chart along with the timestamp
twitter_all <- twitter_all %>% 
  mutate(influence = favorites + 2 * retweets)

influence_plot <- twitter_all %>%
  group_by(timestamp = cut(timestamp, breaks="8 hour")) %>%
             summarise(influence = sum(influence))

influence_plot %>% ggplot(aes(x= timestamp, y=influence, group =1 ))+
  geom_point()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  geom_line(color = "#CC79A7", alpha = 0.7)
  
#### sort out hashtags and find the most influencial ones

hashtags_twitter <- twitter_all %>%
  filter(!is.na(hashtags)) %>%
  count(hashtags) %>%
  arrange(desc(n)) 

hashtags_unnet <- hashtags_twitter %>% unnest_tokens(ht_word,hashtags)

hashtags_unnet <- hashtags_unnet[!duplicated(hashtags_unnet$ht_word), ]
hashtags_unnet <- hashtags_unnet[-1,] ## removed joelosteen which is amost 9k
hashtags_20 <- hashtags_unnet[1:20,]

ggplot(hashtags_20, aes(x= ht_word, y=n))+
  geom_point()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  scale_y_continuous(limits=c(0, 1000))+
  geom_text(aes(label = ht_word), check_overlap = TRUE, vjust = 1.0)


hashtag_28 <- hashtags_twitter <- twitter_all %>%
  filter(day(timestamp) == 28) %>%
  filter(!is.na(hashtags)) %>%
  count(hashtags) %>%
  arrange(desc(n)) %>% unnest_tokens(ht_word,hashtags)

hashtag_28 <- hashtag_28[!duplicated(hashtag_28$ht_word), ]
hashtag_28_20 <- hashtag_28[1:20,]
  
ggplot(hashtag_28_20, aes(x= ht_word, y=n))+
  geom_point()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  scale_y_continuous(limits=c(0, 1000))+
  geom_text(aes(label = ht_word), check_overlap = TRUE, vjust = 1.3)




##
##view missing data
##

library(Amelia)
missmap(twitter_all, main = "Missing values vs observed")

some_twitter <- twitter %>% 
  select(username,text,retweets,date,favorites)
text_twitter_original <- twitter %>% 
  select(text)



##### cluster analysis 
## processing text data into corpus
corp <- VCorpus(DataframeSource((text_twitter_original)))
## pre processing
corp <- tm_map(corp, removeNumbers)
corp <- tm_map(corp, content_transformer(tolower))
corp <- tm_map(corp, content_transformer(rm_twitter_url))

# after using the base funtion, force the format back to tm
# by using plaintextdocument

corp <- tm_map(corp, removeWords, stopwords("english")) 
corp <- tm_map(corp,removePunctuation)
corp <- tm_map(corp, stripWhitespace)
corp <- tm_map(corp, removeWords, c("https*$",
                                    "pic.twitter.*$","pictwitter.*$",
                                    "will","joel","joelosteen",
                                    ))

corpcopy = corp

corp_st <- tm_map(corp, content_transformer(stemDocument))
corp <- tm_map(corp_st, content_transformer(stemCompletion),
               dictionary = corpcopy, lazy=TRUE)


#### stage the data to dtm
## tfidf weighted 


## becaue R aborts over huge data, I chose samples here
twitter_fivek <- twitter_all[sample(nrow(twitter_all), 50000), ]

corpus_twitter_fivek <- Corpus(VectorSource(twitter_fivek$text))
corpus_fivek_clean <- corpus_twitter_fivek %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeWords, c("https.*$",
                              "pic.twitter.*$","pictwitter.*$",
                              "will","joel","joelosteen",
                        "osteen","https","get","can","thats"))
  

tdm <- TermDocumentMatrix(corpus_fivek_clean, 
    control = list(weighting = weightTfIdf)) 
  
tdm_m <- as.matrix(tdm)
tdm_df <- as.data.frame(tdm_m)


freq <- sort(rowSums(tdm_df), decreasing=TRUE)   
head(freq, 14) 

## frequency plot
wf <- data.frame(word=names(freq), freq=freq)
wf <- wf[-c(2,4), ]
fre_plot <- ggplot(subset(wf, freq>1000), aes(x = reorder(word, -freq), y = freq)) +
  geom_bar(stat = "identity") + 
  theme(axis.text.x=element_text(angle=45, hjust=1))+
  labs(x= "most frequent word")
fre_plot

# calculate the frequency of words and sort it by frequency



## finding association with certain words
asso <- findAssocs(tdm, c("christian"), corlimit = 0.05)

## word cloud
set.seed(142)   
wordcloud(names(freq), freq, min.freq=400, scale=c(5, .2),
          colors=brewer.pal(6, "Dark2"))

## cluster analysis
tdm_cluster <- removeSparseTerms(tdm, sparse = 0.98) #### change this number
d <- dist(x=as.matrix(tdm_cluster), method="euclidian")
cluster <- hclust(d,method="ward.D")
cluster
plot(cluster, hang = 0)
rect.hclust(cluster,5)
####
#### even smaller sample to run the principle components

twitter_onek <- twitter_all[sample(nrow(twitter_all), 2000), ]

corpus_twitter_onek <- Corpus(VectorSource(twitter_onek$text))
corpus_onek_clean <- corpus_twitter_onek %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeWords, c("https.*$",
                        "pic.twitter.*$","pictwitter.*$",
                        "will","joel","joelosteen",
                        "osteen","https","get","can","thats"))
tdm_onek <- TermDocumentMatrix(corpus_onek_clean, 
                          control = list(weighting = weightTfIdf)) 
tdm_m_onek <- as.matrix(tdm_onek)
tdm_df_onek <- as.data.frame(tdm_m_onek)

####
#### topic models
####

dtm <- DocumentTermMatrix(corpus_fivek_clean)
rowTotals <- apply(dtm , 1, sum) #Find the sum of words in each Document
dtm.new   <- dtm[rowTotals> 0, ]
install.packages("topicmodels")
library(topicmodels)
lda <- LDA(dtm.new, k = 3) # find 6 topics
term <- terms(lda, 6) # first 6 terms of every topic
term





#### prnciple component analysis
pc <- prcomp(tdm_onek)
plot(pc)
## can find out how many groups by looking at the variances explained
summary(pc)

wss <- (nrow(tdm_df_onek)-1)*sum(apply(tdm_df_onek,2,var))
for (i in 2:15) {
  cat(i)
  wss[i] <- sum(kmeans(tdm_df,
  centers=i)$withinss)
  }
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

m3 <- t(tdm_df_onek) # transpse from term document matriz

k <- 3 # number of clusters
kmeansResult <- kmeans(m3, k)
km <-round(kmeansResult$centers, digits = 3)

for (i in 1:k) {
  cat(i)
  cat(paste("cluster ", i, ": ", sep = ""))
  s <- sort(kmeansResult$centers[i, ], decreasing = T)
  cat(names(s)[1:8], "\n")
}

library(fpc)
distance <- dist(tdm_df_onek, method ="euclidian")
kfit <- kmeans(distance, 2)
clusplot(as.matrix(distance), kfit$cluster, 
         color=T, shade= T, labels= 2, lines=0)
fit$cluster[kfit$cluster == 2]
# choose to show the cluster elements


