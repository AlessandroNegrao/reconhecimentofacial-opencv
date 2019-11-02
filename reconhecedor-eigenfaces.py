import cv2

detectorFace = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("classificadorEigen.yml")
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

camera = cv2.VideoCapture(0)

while(True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza,
                                                    scaleFactor=1.5,
                                                    minSize=(100, 100))
    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y +a, x:x +l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (255, 0, 0), 2)
        id, confianca = reconhecedor.predict(imagemFace)
        if id == 1:
           nome = 'Alessandro'
        #elif id == 2:
         #   nome = 'Jean'
        elif id == 3:
            nome = 'Daniel'
        #elif id == 4:
         #   nome = 'Christian'
        elif id == 5:
            nome = 'Andreson'
        elif id == 6:
            nome = 'Ana Paula'
        elif id == 7:
            nome = 'Gabriel'
        elif id == 8:
            nome = 'Elias'
        elif id == 9:
            nome = 'Leanderson'
        elif id == 10:
            nome = 'Thiago S.'
        elif id == 11:
            nome = 'Rickson'
        elif id == 12:
            nome = 'Gabiel'
        elif id == 13:
            nome = 'Harley'
        elif id == 14:
            nome = 'Joao'
        elif id == 15:
            nome = "Alessandro 2"
        else:
            nome = "Undefined"
            #str(id)
        cv2.putText(imagem, nome, (x, y + (a + 30)), font, 2, (0, 0, 255))

    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()