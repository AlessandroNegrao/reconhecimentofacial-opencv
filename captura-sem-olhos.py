import cv2
import numpy as np

#classificador utilizado para a detecção de faces
classificador = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")

#Conexão com a web cam, no caso por ser apenas uma, com id 0
camera = cv2.VideoCapture(0)

#Variáveis para captura de amostras (fotos dos rostos
amostra = 1
numeroAmostras = 40
largura = 220
altura = 220
print("Capturando as faces")

id = input("Digite seu identificador: ")

#Iniciar atividades da web cam
while (True):
    conectado, imagem = camera.read()

    #converter as imagens recebidas em escala de cinza
    #imagem = imagem colorida receptada pela web cam
    #imagemCinza = imagem em escala de cinza, convertida para a detecção

    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    #print(np.average(imagemCinza))
    #Variável que armazena temporariamente as faces detectadas
    facesDetectadas = classificador.detectMultiScale(imagemCinza,
                                                     scaleFactor=1.5,
                                                     minSize=(100, 100))

    #Os parâmetros x, y, l e a estão na variável facesDetectadas. 'X' e 'y' é onde começa e onde termina uma face.
    # 'l' é a largura e 'a' a altura. A intenção é produzir um retângulo em volta da face detectada
    for (x, y, l, a) in facesDetectadas:

        #Parâmetros do rectangle
        #cv2.rectangle(imagem captada, (ponto de inicio, ponto de fim), (inicio em largura, fim em altura),
        # (RGB pro quadrado), (tamanho daborda))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (255, 0, 0), 2)
        #waitKey aguarda um comando, o hexadecimal define que o q será usado para armazenas as imagens,
        # de acordo com o código abaixo
        if cv2.waitKey(1) & 0xFF == ord('a'):
            #Verificação de luminosidade em imagem. Precisa ser maior que 110
            if np.average(imagemCinza) > 110:

                #Aqui a imagem está sendo capturada
                imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                #criação da imagem capturada redimencionada
                cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace)
                print("[A foto " + str(amostra) + " da amostra " + str(id) + " foi capturada com sucesso!]")
                amostra += 1
    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    #Ao atingir o numero de amostras, o comando deve dar um break
    if (amostra >= numeroAmostras + 1):
        break

print("Faces capturadas com sucesso!")
camera.release()
cv2.destroyAllWindows()