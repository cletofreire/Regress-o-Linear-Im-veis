#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importando as  bibliotecas
import pandas as pd
import seaborn as sns


# Importando os dados 
df = pd.read_csv('C:/Users/cleto/Documents/DATA-SCIENCE/Aulas/Machine/Regressão/Alura/Aula2/reg-linear-II/Dados/dataset.csv', sep=';')
df.head()

# Observado a tipo de dados 
df.info()


#  ESTATISTICAS DESCRITIVAS


# Um apanhado estatístico dos dados
df.describe().round(3)

#Criando uma matriz de correlação
df.corr()
# foi observado que há correção positiva entre a área e negativas entre a distancia da praia  e pouco relevante para dis_farmacia.


# Observando graficamente os valores
ax = sns.boxplot(data = df['Valor'], orient ='h', width = 0.5 )
ax.figure.set_size_inches(20, 6)
ax.set_title('Preço dos Imóveis', fontsize=20)
ax.set_xlabel('Reais', fontsize=16)
ax

# Observando a distribuição de frequencia

ax = sns.histplot(df['Valor'], kde= True, bins = 45  )
ax.figure.set_size_inches(20, 6)
ax.set_title('Distribuição de Frequências', fontsize=20)
ax.set_xlabel('Preço dos Imóveis (R$)', fontsize=16)
ax


#PAIRPLOT
# Estimando uma reta de regressão para observar +- como seria o danado da tendencia.
ax = sns.pairplot(df, y_vars ='Valor',x_vars = ['Dist_Praia','Dist_Farmacia','Area'], height = 10, kind = 'reg')
ax.fig.suptitle('Dispersão entre as Variáveis', fontsize=20, y=1.05)
ax



#TRANSFORMANDO OS DADOS

# Como foi observado anteriormente a distribuição de frequencia dos dados está com a cauda a direita, um dos pressupostos da regressão linear é a normalidade das distribuições
# sendo necessário que ambos estejam centralizados ou igualmente distribuídos.


# Optei pela transformação logarímitca
df['log_Valor'] = np.log(df['Valor'])
df['log_Area'] = np.log(df['Area'])
df['log_Dist_Praia'] = np.log(df['Dist_Praia'] +1 )
df['log_Dist_Farmacia'] = np.log(df['Dist_Farmacia'] +1)


# Observando a distribuição por meio do histograma, foi possível observar que está próximo de uma relação normal.
sns.histplot(df['log_Valor'])
ax.figure.set_size_inches(20, 6)


# Gráficos de dispersão 
ax = sns.pairplot(df, y_vars ='log_Valor',x_vars =['log_Area','log_Dist_Praia','log_Dist_Farmacia'], kind = 'reg' )
ax.figure.set_size_inches(20, 6)
ax.fig.suptitle('Dispersão entre as Variáveis Transformadas', fontsize=20, y=1.05)
ax



# CRIANDO O MODELO PREDITIVO


# Importando a biblioteca
from sklearn.model_selection import train_test_split

# Armazenando a variável y
y = df['log_Valor' ]
#Armazenando a variavel x
X = df [['log_Dist_Farmacia','log_Dist_Praia', 'log_Area']]

#Criando os dataset de treino e de teste

X_test,X_train,y_test,y_train = train_test_split(X,y, test_size = 0.20, random_state = 2811)


####################################################################
# Importando Statsmodels
import statsmodels.api as sm

# Paraque o stasmodel funcione é necessário adicionar uma constante no data frame de variáveis explicativas do statsmodel (Constante 1)
X_train_constante = sm.add_constant(X_train)
X_train_constante

# O modelo de treino,  mas agora com o x de treino constante.
modelo_stats = sm.OLS(y_train, X_train_constante, hasconst= True).fit()

# Observando o resultado, que teve o R² bem satisfatório.
print(modelo_stats.summary())


# Observei que a distancia da farmácia não é tão explicativa assim para o modelo então decidi retirar
X = df [['log_Dist_Praia', 'log_Area']]

#Recriando os modelos de treino e teste, sem a variável dis_farmacia, melhor ajustado
X_train, X_test,y_train,y_test = train_test_split(X,y, random_state = 2811, test_size = 0.20)


X_train_constante = sm.add_constant(X_train)

modelo_stats = sm.OLS(y_train, X_train_constante, hasconst= True).fit()

print(modelo_stats.summary())
################################################################


# IMPORTANDNO LINEARREGRESSION

from sklearn.linear_model import LinearRegression
from sklearn import metrics

#trazendo a classe linear regression
modelo = LinearRegression()

# Utilizando o modelo de treino
modelo.fit(X_train, y_train)


# Observando o valor R² Do modelo de treino R² = 80%
print('R² ={}'.format(modelo.score(X_train,y_train).round(3)))

#Agora incia  o modelo de teste

y_previsto = modelo.predict(X_test)

# Obtendo o valor do coefiente de determinação (R²) do modelo de teste , R² = 0.79

print('R² = %s' % metrics.r2_score(y_test,y_previsto).round(3))


# GERANDO UM MODELO DE PREVISÃO 

# Captando os dados , presentes no X_test
entrada  = X_test[4:11]

# Gerando uma previsão pontual dos mesmo
modelo.predict(entrada)

# Desfazendo o a transformação logarítimica,  A função np.exp ela DESFAZ A FUNÇÃO DE LOG
np.exp(modelo.predict(entrada)[0])



# Com esse modelo simplificado é possível adicionar valores de distancia e área e prever o valor do imóvel
# A exemplo disto eu adicionei uma área de 250 m² e uma distancia  curta da praia, e o valor do imóvel foi de 277k

#####  ### CUIDADO, VOCÊ DEVE COLOCAR NA ORDEM QUE ESTÃO ORGANIZADAS AS SUA TABELA NO MEU CASO , PRIMEIRO DIST PRAIA DEPOIS AREA
# SE NÃO DA ERRO.
Area = 250
Dist_Praia = 1
entrada = [[np.log(Dist_Praia + 1), np.log(Area)]]   # AQUI TEM QUE TA EM ORDEM

np.exp(modelo.predict(entrada))


#  Obtendo o valor do intecepto
modelo.intercept_
# o valor do intercepto neste modelo pode ser traduzido basicamente como o valor do imóvel sem a influencia dos fatores distância da praia e área.
np.exp(modelo.intercept_)

#  Os valores dos coeficientes de regressão que são a distancia da praia e área
modelo.coef_   
# É possível obter o


# colocando tudo em um DF só pra organizar
X.columns # observando o nome das colunas para não errar
index = ['Intercepto', 'Distancia até a praia','Área'] # criando uma lista para organizar a variáveis do modelo


# Criando um DF
pd.DataFrame(data = np.append(modelo.intercept_,modelo.coef_), index = index)

#RESULTADO: # a cada metro de distancia da praia (km) , diminui 0.49 do valor
# acada m² de área aumenta 1.05






