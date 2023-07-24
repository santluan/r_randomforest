# Projeto Random Forest

# Definindo o diretório de trabalho
setwd()

# Carregando pacotes necessários
library(Amelia) # para tratar valores ausentes
library(caret) # construcao de modelos ML e processamento de dados 
library(ggplot2) # vizualizacao de dados
library(dplyr) # tratamento de dados
library(reshape) # modificar o formato dos dados
library(randomForest) # pacotes para trabalhar com ML
library(e1071) # pacotes para trabalhar com ML
library(readr)

# Carregando base de dados
clientes <- read_csv("Clients.csv")

# Visualizando dados e sua estrutura
dim(clientes)
str(clientes, give.attr = F)
summary(clientes)

#### Análise Exploratóra, Manipulação, Limpeza e Tratamento de Dados ####

### Verificando valores ausentes e removendo do dataset
# Para visualizar valores ausentes
missmap(clientes, main = "Valores Missing Observados")

# contagem de valores ausentes
sapply(clientes, function(x) sum(is.na(x))) 

# limpando a base
clientes <- clientes %>%
  mutate(...1 = NULL,
         education = NULL) %>%
  tidyr::drop_na()

head(clientes)

# Transformando a variável de idade para o tipo fator com faixas etárias
# clientes$age <- cut(clientes$age,
#                            c(0, 25, 60, 100),
#                            labels = c("Jovem",
#                                       "Adulto",
#                                       "Idoso"))

# alterando a variável dependente para o tipo fator
str(clientes$BAD)

clientes$BAD <- as.factor(clientes$BAD)

# Total de adimplentes vs inadimplentes
table(clientes$BAD)

# Em porcentagens
prop.table(table(clientes$BAD))

# Plot da distribuição usando ggplot2
clientes %>%
  ggplot(aes(x = BAD, y = length(BAD))) +
  geom_bar(stat = "identity") +
  labs(x = "BAD", y = "Quantidade") +
  theme_bw()

# Set seed
set.seed(1234)

### Amostragem estratificada
# Seleciona as linhas de acordo com a variável inadimplente como strata

indice <- createDataPartition(clientes$BAD, p = 0.75, list = F)
dim(indice)

# Definindo os dados de treinamento como subconjunto da base de dados original
# atraves dos numeros indices de linha

dados_treino <- clientes[indice,]
dim(dados_treino)
table(dados_treino$BAD)

prop.table(table(dados_treino$BAD))
prop.table(table(clientes$BAD))

compare <- cbind(Treinamento = prop.table(table(dados_treino$BAD)),
                 Original = prop.table(table(clientes$BAD)))

melt_compare <- melt(compare)

melt_compare %>%
  ggplot(aes(x = X1, y = value)) +
  geom_bar(aes(fill = X2), stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Definindo os dados de teste
dados_teste <- clientes[-indice,]
dim(dados_teste)

#### Modelo de Machine Learning ####

modelo_v1 <- randomForest(BAD ~ ., data = dados_treino, ntree = 100)
modelo_v1

# Avaliando o modelo
plot(modelo_v1)

# Previsoes com dados de teste
previsoes_v1 <- predict(modelo_v1, dados_teste)

# Matriz de confusao (confusion matrix)
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$BAD, positive = "1")
cm_v1

# Calculando Precision, Recall e F1-Score, métricas de avaliacao do modelo preditivo
y <- dados_teste$BAD
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision

recall <- sensitivity(y_pred_v1, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

#### Versao 2 - Balanceamento de classe - undersampling ####

table(dados_treino$BAD)

dados_balanceado <- dados_treino %>%
  group_by(BAD) %>%
  sample_n(5925) %>%
  ungroup()

table(dados_balanceado$BAD)

### Testando modelo balanceado

modelo_v2 <- randomForest(BAD ~ ., dados_balanceado)
modelo_v2

# avaliando modelo
plot(modelo_v2)

# previsoes
previsoes_v2 <- predict(modelo_v2, dados_teste)
previsoes_v2

cm_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$BAD, positive = "1")
cm_v2

# Calculando Precision, Recall e F1-Score, métricas de avaliacao do modelo preditivo
y <- dados_teste$BAD
y_pred_v2 <- previsoes_v2

precision <- posPredValue(y_pred_v2, y)
precision

recall <- sensitivity(y_pred_v2, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

#### Avaliando a importancia das variáveis preditoras para as previsoes ####

varImpPlot(modelo_v1)
varImpPlot(modelo_v2)

# obtendo a importancia para as variaveis do modelo
imp_var <- importance(modelo_v2)
varImportance <- data.frame(Variables = row.names(imp_var),
                            Importance = round(imp_var[, 'MeanDecreaseGini'], 2))


# Criando o rank de variaveis baseado na importancia
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

# Usando ggplot2 para visualizar a importancia relativa das variaveis
rankImportance %>%
  ggplot(aes(x = reorder(Variables, Importance),
             y = Importance,
             fill = Importance)) +
  geom_bar(stat = 'identity') +
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust = 0,
            vjust = 0.55,
            size = 4,
            colour = 'red') +
  labs(x = 'Variables') +
  coord_flip()


#### Versao 3 do Modelo - Testando modelo com as variaveis mais relevantes ####

modelo_v3 <- randomForest(BAD ~ 
                            age + personalNetIncome + professionCode + monthsInResidence +
                            monthsInTheJob + bestPaymentDay,
                          data = dados_balanceado)

# avaliando modelo
plot(modelo_v3)

# previsoes
previsoes_v3 <- predict(modelo_v3, dados_teste)
previsoes_v3

cm_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$BAD, positive = "1")
cm_v3
