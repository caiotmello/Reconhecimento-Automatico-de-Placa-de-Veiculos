import numpy as np
import cv2 as cv
import os
import math
import tkinter
from DetectVehiclePlate import DetectVehiclePlate
from OCR import Ocr

#Acha a placa dentro do cenario da placa
plate = DetectVehiclePlate('./Imagens/22-ok.jpg',False)
placaDetectada = plate.detectaPlaca()

detectCaracter = Ocr(False)
frase = detectCaracter.reconhecimentoOCR(placaDetectada)
print(frase)

# Realiza um filtro nos caracteres obtidos 
# eliminando possiveis ruidos reconhecidos
if len(frase) > 0:
    texto = detectCaracter.removerChars(frase)
else:
    texto = "Reconhecimento Falho"

print(texto)
#Abre uma janela com os algoritmos da placa
janela = tkinter.Tk()
tkinter.Label(janela, text=texto, font=("Helvetica", 50)).pack()
janela.mainloop()

cv.waitKey(0)
cv.destroyAllWindows()


