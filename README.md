from sklearn.preprocessing import LabelEncoder

codificador = LabelEncoder()

tabela["profissao"] = codificador.fit_transform(tabela["profissao"])

tabela["mix_credito"] = codificador.fit_transform(tabela["mix_credito"])

tabela["comportamento_pagamento"] = codificador.fit_transform(tabela["comportamento_pagamento"])

display(tabela.info())
#Passo 2. Preparar a base de dados para a inteligencia artificial
#float - numero com casa decimal 
#int -  inteiro 
#object - coluna de texto
# profissao
    # Engenheiro -1 
    # advogado -2
    # cientista -3
    # ator -4
    # mecanico -5

# mix-credito
# comportamento_pagamento

from sklearn.preprocessing import LabelEncoder

codificador = LabelEncoder()

tabela["profissao"] = codificador.fit_transform(tabela["profissao"])

tabela["mix_credito"] = codificador.fit_transform(tabela["mix_credito"])

tabela["comportamento_pagamento"] = codificador.fit_transform(tabela["comportamento_pagamento"])

display(tabela.info())
# a primeira primeira pergunta que se tem que fazer é = "qual é a coluna da base de dados que eu quero ser capaz de prever? a primeira coluna é a score de credito y"
y = tabela["score_credito"]
# a segunda é todas as outras que vamos chamar de X
x = tabela.drop(columns=["score_credito", "id_cliente"])

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# 3 criar o modelo de IA

#Arvore de decisao -> RandomForest
#KNN- vizinhos proximos (nears neighbors)
#importar a inteligencia artificial 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#criar a inteligencia artificial
modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

#treinar a inteligencia artificial
modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino) 

# 4 Escolher qual melhor modelo de IA
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)

#acuracia  = acuracy

from sklearn.metrics import accuracy_score

display(accuracy_score(y_teste, previsao_arvoredecisao))
display(accuracy_score(y_teste, previsao_knn))

# 5 Usar a nossa Ia para definir o score de credito do cliente 

novos_clientes = pd.read_csv("novos_clientes.csv")
display(novos_clientes)

novos_clientes["profissao"] = codificador.fit_transform(novos_clientes["profissao"])

novos_clientes["mix_credito"] = codificador.fit_transform(novos_clientes["mix_credito"])

novos_clientes["comportamento_pagamento"] = codificador.fit_transform(novos_clientes["comportamento_pagamento"])

previsao = modelo_arvoredecisao.predict(novos_clientes)
display(previsao)
