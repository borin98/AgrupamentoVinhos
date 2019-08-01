import pandas as pd
import matplotlib.pylab as plt

from minisom import MiniSom as MS
from sklearn.preprocessing import MinMaxScaler
from pylab import plot, pcolor, colorbar, show, savefig

"""
Leitura e criação do dataframe

    :return -> retorna o modelo treinado e a matriz de ativação dos neurônios

"""
def preProcessing (  ) :

    df = pd.read_csv ( "wines.csv" )

    # atributos previsores, retirando que tipo de classe é
    x = df.iloc[:,1:14].values
    y = df.iloc[:,0].values

    # conversão do valor y
    y[y == 1] = 0
    y[y == 2] = 1
    y[y == 3] = 2

    # normalizando os valores
    normalizador = MinMaxScaler(feature_range=(0,1))
    x = normalizador.fit_transform(x)

    # quantidade de linhas que o mapa vai ter
    som = MS(x=8, y=8, input_len=13,sigma=1,learning_rate=0.5,random_seed=2)
    som.random_weights_init(x)
    som.train_random(data=x,num_iteration=100,verbose=True)

    # matriz de atvivação dos neurônios
    q = som.activation_response(x)

    return som, q, x, y

"""

Função que faz a vizualização dos dados

"""

def vizuDados(som, x, y) :

    markers = ["o", "o", "o"]
    color = ["r", "g", "b"]

    for i, j in enumerate(x) :

        w = som.winner(j)

        plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
             markerfacecolor = "None", markersize = 10,
             markeredgecolor = color[y[i]], markeredgewidth = 2)


    show()
    # usar o savefig e retirar o show, pois se não buga
    # savefig("grafico.pdf")

def main(  ) :

    som, _, x, y = preProcessing()

    # vizualizando a imagem antes do plot final
    pcolor(som.distance_map().T)
    colorbar()
    #plot()
    #plt.show()
    #plt.savefig("colorBar.pdf")

    vizuDados(som=som, x=x, y=y)

if __name__ == '__main__':
    main()