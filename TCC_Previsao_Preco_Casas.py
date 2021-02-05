"""
PONTIFÍCIA UNIVERSIDADE CATÓLICA DE MINAS GERAIS
NÚCLEO DE EDUCAÇÃO A DISTÂNCIA
Pós-graduação Lato Sensu em Ciência de Dados e Big Data

Heiji Inuzuka

Previsão de preços de casas com base nas características do imóvel e crimes por CEP

Trabalho de Conclusão de Curso apresentado ao Curso de Especialização em Ciência de Dados e Big Data 
como requisito parcial à obtenção do título de especialista.
"""

import pandas as pd
import numpy as np


"""
Leitura das bases de dados
"""

dados_imoveis = pd.read_csv('home_data.csv')

ceps = pd.read_csv('US.txt', delimiter="\t", header=None, names=['country code',
                                                                 'zipcode',
                                                                 'place name',
                                                                 'admin name1',
                                                                 'admin code1',
                                                                 'admin name2',
                                                                 'admin code2',
                                                                 'admin name3',
                                                                 'admin code3',
                                                                 'latitude',
                                                                 'longitude',
                                                                 'accuracy'
                                                                 ])

crimes = pd.read_csv('King_County_Sheriff_s_Office_-_Incident_Dataset.csv')




"""
Análise e pré-processamento das bases de dados
"""

# Verificando valores nulos
dados_imoveis.isnull().sum()
ceps.isnull().sum()
crimes.isnull().sum()

# Removendo as linhas do DataFrame crimes, cujos valores de zip são nulos
crimes = crimes.dropna(axis=0, subset=['zip'])

# Juntando o DataFrame dados_imoveis com o ceps
dados_imoveis_com_dados_cep = pd.merge(dados_imoveis, ceps, on = ['zipcode'], how = 'left')

# Agrupando por "admin name1" e "admin name2"
dados_imoveis_com_dados_cep.groupby('admin name1').agg('count')
dados_imoveis_com_dados_cep.groupby('admin name2').agg('count')

# Retirando os dados dos imóveis da região Snohomish no dataset original
linhas_remover = dados_imoveis_com_dados_cep.loc[dados_imoveis_com_dados_cep['admin name2']=='Snohomish']
dados_imoveis_King = dados_imoveis.drop(linhas_remover.index)

# Retirando as colunas de metragem de 2015
dados_imoveis_King = dados_imoveis_King.drop(['sqft_living15', 'sqft_lot15'], axis=1)

# Pegando as datas dos anúncios e classificando em ordem
periodo_dados_imoveis = dados_imoveis_King['date'].sort_values()

# Criando uma coluna no Dataframe crimes, com o ano em que o crime ocorreu, agrupando os crimes por ano
crimes['ano'] = crimes['incident_datetime'].apply(lambda x: str(x)[6:10])
agrupamento_crimes = crimes.groupby('ano').agg('count')['case_number']

# Filtrando os crimes de 2018 e 2019
crimes_2018_2019 = crimes[(crimes['ano']=='2018') | (crimes['ano']=='2019')]

# Criando coluna contador para fazer soma
crimes_2018_2019['contador'] = 1

# Agrupando os crimes de 2018 e 2019 por zipcode
crimes_por_cep = pd.pivot_table(crimes_2018_2019, index=['zip'], values=['contador'], columns=['incident_type'], aggfunc=[np.sum])
crimes_por_cep.columns=['Arson',
                        'Assault',
                        'Assault with Deadly Weapon',
                        'Breaking & Entering',
                        'Community Policing',
                        'Death',
                        'Disorder',
                        'Drugs',
                        'Fire',
                        'Homicide',
                        'Kidnapping',
                        'Liquor',
                        'Missing Person',
                        'Other',
                        'Other Sexual Offense',
                        'Property Crime',
                        'Robbery',
                        'Sexual Assault',
                        'Theft',
                        'Theft from Vehicle',
                        'Theft of Vehicle',
                        'Traffic',
                        'Vehicle Recovery',
                        'Weapons Offense']

# Criando nova coluna com zipcode para fazer o merge
crimes_por_cep['zipcode'] = crimes_por_cep.index

# Retirando a linha com dado V4W 2W1, que não representa um zipcode
crimes_por_cep = crimes_por_cep.drop(['V4W 2W1'])

# Transformando a coluna zipcode de crimes_por_cep em inteiro, para ficar igual ao zipcode dos dados imóveis
crimes_por_cep['zipcode'] = pd.to_numeric(crimes_por_cep['zipcode'], downcast= 'integer')

# Enriquecendo a base inicial com a contagem de crimes por cep
dados_imoveis_enriquecido = pd.merge(dados_imoveis_King, 
                                     crimes_por_cep, 
                                     on = ['zipcode'], 
                                     how = 'left')

# Verificando a quantidade de valores nulos por coluna
dados_imoveis_enriquecido.isnull().sum()

# Substituindo os valores de case_number de nan para zero (0)
dados_imoveis_enriquecido = dados_imoveis_enriquecido.fillna(0)

# Criando coluna com a soma de todos os crimes
dados_imoveis_enriquecido['soma_crimes'] = dados_imoveis_enriquecido.iloc[:,19:43].sum(axis=1)


# Analisando matriz de correlação e mapa de calor
import seaborn as sns
correlacao = dados_imoveis_enriquecido.corr()
sns.heatmap(correlacao, xticklabels=correlacao.columns, yticklabels=correlacao.columns)



"""
Grupo 1: Dados originais
"""
base_ML = dados_imoveis_enriquecido
x = base_ML.iloc[:, 3:19].values
y = base_ML.iloc[:, 2].values
n_camadas_ocultas = 9 # variar entre 9 e 12


"""
Grupo 2: Dados originais + soma de crimes
"""
base_ML = dados_imoveis_enriquecido.drop(dados_imoveis_enriquecido.iloc[:,19:43],axis=1)
x = base_ML.iloc[:, 3:20].values
y = base_ML.iloc[:, 2].values
n_camadas_ocultas = 12 # variar entre 9 e 12


"""
Grupo 3: Dados originais + crimes por tipo
"""
base_ML = dados_imoveis_enriquecido
x = base_ML.iloc[:, 3:43].values
y = base_ML.iloc[:, 2].values
n_camadas_ocultas = 26 # variar entre 20 e 26






# Importações para utilização nos modelos de machine learning de regressão
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error





"""
REGRESSÃO LINEAR MÚLTIPLA

"""

# Importando pacote de Regressão Linear do sklearn
from sklearn.linear_model import LinearRegression

# Criando matriz de zeros para armazenar os 100 resultados
resultados_100 = np.zeros((100,3))

# Criando loop para rodar 100 vezes variando i de 1 a 100
for i in range(100):    
    
    # Dividindo as variáveis x e y em 70% para treinamento e 30% para teste conforme semente de random_state
    x_treinamento, x_teste, y_treinamento, y_teste = train_test_split (x, 
                                                                       y, 
                                                                       test_size = 0.3, 
                                                                       random_state = i)

    # Fazendo o treinamento para gerar a equação do modelo
    regressor = LinearRegression()
    regressor.fit(x_treinamento, y_treinamento)
    
    # Calculando a porcentagem de acerto na base de treinamento com base no modelo gerado
    score_treinamento = regressor.score (x_treinamento, y_treinamento)
    
    # Calculando a porcentagem de acerto na base de teste com base no modelo gerado
    score_teste = regressor.score(x_teste, y_teste)
    
    # Calculando as previsões da base de teste
    previsoes = regressor.predict(x_teste)
    
    # Calculando o erro médio absoluto entre as previsões e os valores reais da base de teste
    mae = mean_absolute_error (y_teste, previsoes)
    
    # Colocando os resultados obtidos na variável resultados_100        
    resultados_100[i-1][0] = score_treinamento
    resultados_100[i-1][1] = score_teste
    resultados_100[i-1][2] = mae
    
# Calculando a média dos 100 resultados para score_treinamento, score_teste e mae
media_100 = resultados_100.mean(axis=0)
    




"""
REGRESSÃO POLINOMIAL

"""

# Importando pacotes de Regressão Linear e Polinomial do sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Criando matriz de zeros para armazenar os 100 resultados
resultados_100 = np.zeros((100,3))

# Criando loop para rodar 100 vezes variando i de 1 a 100
for i in range(100):
    
    # Dividindo as variáveis x e y em 70% para treinamento e 30% para teste conforme semente de random_state
    x_treinamento, x_teste, y_treinamento, y_teste = train_test_split (x, 
                                                                       y, 
                                                                       test_size = 0.3, 
                                                                       random_state = i)
    
    # Gerando outras colunas conforme grau do polinômio    
    poly = PolynomialFeatures(degree = 2)  # variar entre 2 e 3    
    x_treinamento_poly = poly.fit_transform(x_treinamento)
    x_teste_poly = poly.transform(x_teste)


    # Fazendo o treinamento para gerar a equação do modelo
    regressor = LinearRegression()
    regressor.fit(x_treinamento_poly, y_treinamento)
    
    # Calculando a porcentagem de acerto na base de treinamento com base no modelo gerado
    score_treinamento = regressor.score (x_treinamento_poly, y_treinamento)
    
    # Calculando a porcentagem de acerto na base de teste com base no modelo gerado
    score_teste = regressor.score(x_teste_poly, y_teste)

    # Calculando as previsões da base de teste
    previsoes = regressor.predict(x_teste_poly)
    
    # Calculando o erro médio absoluto entre as previsões e os valores reais da base de teste
    mae = mean_absolute_error (y_teste, previsoes)    
    
    # Colocando os resultados obtidos na variável resultados_100 
    resultados_100[i-1][0] = score_treinamento
    resultados_100[i-1][1] = score_teste
    resultados_100[i-1][2] = mae
        
# Calculando a média dos 100 resultados para score_treinamento, score_teste e mae
media_100 = resultados_100.mean(axis=0)




"""
REGRESSÃO COM ÁRVORES DE DECISÃO

"""

# Importando pacote de Regressão por árvores de decisão do sklearn
from sklearn.tree import DecisionTreeRegressor

# Criando matriz de zeros para armazenar os 100 resultados
resultados_100 = np.zeros((100,3))

# Criando loop para rodar 100 vezes variando i de 1 a 100
for i in range(100):    
    
    # Dividindo as variáveis x e y em 70% para treinamento e 30% para teste conforme semente de random_state
    x_treinamento, x_teste, y_treinamento, y_teste = train_test_split (x, 
                                                                       y, 
                                                                       test_size = 0.3, 
                                                                       random_state =i)
    
    # Fazendo o treinamento para gerar a equação do modelo
    regressor = DecisionTreeRegressor(max_depth=12) #Variar entre None, 10, 11 e 12 
    regressor.fit(x_treinamento, y_treinamento)
    
    # Calculando a porcentagem de acerto na base de treinamento com base no modelo gerado
    score_treinamento = regressor.score (x_treinamento, y_treinamento)
    
    # Calculando a porcentagem de acerto na base de teste com base no modelo gerado
    score_teste = regressor.score(x_teste, y_teste)

    # Calculando as previsões da base de teste
    previsoes = regressor.predict(x_teste)
    
    # Calculando o erro médio absoluto entre as previsões e os valores reais da base de teste
    mae = mean_absolute_error (y_teste, previsoes)
        
    # Colocando os resultados obtidos na variável resultados_100
    resultados_100[i-1][0] = score_treinamento
    resultados_100[i-1][1] = score_teste
    resultados_100[i-1][2] = mae
    
# Calculando a média dos 100 resultados para score_treinamento, score_teste e mae        
media_100 = resultados_100.mean(axis=0)




"""
REGRESSÃO COM RANDOM FOREST

"""

# Importando pacote de Regressão por Random Forest do sklearn
from sklearn.ensemble import RandomForestRegressor

# Criando matriz de zeros para armazenar os 100 resultados
resultados_100 = np.zeros((100,3))

# Criando loop para rodar 100 vezes variando i de 1 a 100
for i in range(100):   
    
    # Dividindo as variáveis x e y em 70% para treinamento e 30% para teste conforme semente de random_state
    x_treinamento, x_teste, y_treinamento, y_teste = train_test_split (x, 
                                                                       y, 
                                                                       test_size = 0.3, 
                                                                       random_state =i)
    
     # Fazendo o treinamento para gerar a equação do modelo
    regressor = RandomForestRegressor(n_estimators = 50) # variar entre 10, 50 e 100
    regressor.fit(x_treinamento, y_treinamento)
    
    # Calculando a porcentagem de acerto na base de treinamento com base no modelo gerado
    score_treinamento = regressor.score (x_treinamento, y_treinamento)
    
    # Calculando a porcentagem de acerto na base de teste com base no modelo gerado
    score_teste = regressor.score(x_teste, y_teste)

    # Calculando as previsões da base de teste
    previsoes = regressor.predict(x_teste)
    
    # Calculando o erro médio absoluto entre as previsões e os valores reais da base de teste
    mae = mean_absolute_error (y_teste, previsoes)    
    
    # Colocando os resultados obtidos na variável resultados_100
    resultados_100[i-1][0] = score_treinamento
    resultados_100[i-1][1] = score_teste
    resultados_100[i-1][2] = mae
 
# Calculando a média dos 100 resultados para score_treinamento, score_teste e mae        
media_100 = resultados_100.mean(axis=0)




"""
REGRESSÃO COM REDES NEURAIS MULTICAMADAS

"""

# Transformando o formato do array da variável y para poder escalonar
y = dados_imoveis_enriquecido.iloc[:, 2:3].values

# Importando pacotes de Regressão de rede neural e de escalonamento do sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Escalonando as variáveis x e y
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# Criando matriz de zeros para armazenar os 100 resultados
resultados_100 = np.zeros((100,3))

# Criando loop para rodar 100 vezes variando i de 1 a 100
for i in range(100):    
    
    # Dividindo as variáveis x e y em 70% para treinamento e 30% para teste conforme semente de random_state
    x_treinamento, x_teste, y_treinamento, y_teste = train_test_split (x, 
                                                                       y, 
                                                                       test_size = 0.3, 
                                                                       random_state = i)
    
    # Fazendo o treinamento para gerar a equação do modelo
    regressor = MLPRegressor(hidden_layer_sizes = (n_camadas_ocultas,n_camadas_ocultas))
    regressor.fit(x_treinamento, y_treinamento)
    
    # Calculando a porcentagem de acerto na base de treinamento com base no modelo gerado
    score_treinamento = regressor.score (x_treinamento, y_treinamento)
    
    # Calculando a porcentagem de acerto na base de teste com base no modelo gerado
    score_teste = regressor.score(x_teste, y_teste)

    # Calculando as previsões da base de teste
    previsoes = regressor.predict(x_teste)
    
    # Revertendo o escalonamento de y_teste e previsões para a variável mae ficar na mesma escala dos outros modelos
    y_teste = scaler_y.inverse_transform(y_teste)
    previsoes = scaler_y.inverse_transform(previsoes)

    # Calculando o erro médio absoluto entre as previsões e os valores reais da base de teste
    mae = mean_absolute_error (y_teste, previsoes)
        
    # Colocando os resultados obtidos na variável resultados_100
    resultados_100[i-1][0] = score_treinamento
    resultados_100[i-1][1] = score_teste
    resultados_100[i-1][2] = mae
        
# Calculando a média dos 100 resultados para score_treinamento, score_teste e mae
media_100 = resultados_100.mean(axis=0)
