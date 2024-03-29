import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

#Percorre as páginas e encontra os ids e as imagens referentes a eles
#Aprendizagem de máquina supervisionada
def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    # print(caminhos)
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        print(id)
        ids.append(id)
        faces.append(imagemFace)
        #imagemFace = cv2.imread(caminhoImagem)
        #cv2.imshow("Face", imagemFace)
        #cv2.waitKey(10)
    return np.array(ids), faces

ids, faces = getImagemComId()
print(ids)
#print(faces)
print("Treinando...")

#Execução do treinamento do eigenface
eigenface.train(faces, ids)

#Armazena os dados do treinamento feito acima
eigenface.write('classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print('Treinamento realizado!')