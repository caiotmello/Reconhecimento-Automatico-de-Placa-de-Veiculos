import numpy as np
import cv2 as cv
import os
import math
import pytesseract  as tesseract
import tkinter
from PIL import Image

class Ocr:
    placa = []
    showsteps = False
    imagemFrase=[]
    textPlate=[]
    
    #Construtor
    def __init__(self,steps):
        self.showsteps = steps
        
    def x_cord_contour(self,contours):
        #retorna a coordenada X para o contorno 
        M = cv.moments(contours)
        if M['m00'] == 0:
            return 1
        
        return (int(M['m10']/M['m00']))
        
    def verificaChar(self,largura,altura):
        #Verifica os contornos que possue o mesmo tamanho dos caracteres de uma placa
        error =0.3;
        aspect =0.61290 # Placa Brasil 38cmx62cm aspect 38/62 = 0.61290
                                 # Placa Europa size 45.0f/77.0f aspect 3.0769

        #Min e Max area. Todos os outros tamanhos são discartados
        minPerimetro = 85 # area min #200
        maxPerimetro = 600 #area maxima

        #Pega Areas que respeitao o ratio
        rmin = 0.20         #O minimo é do caracter I e 1 qu etem dimensao 13x62
        rmax = aspect + aspect*error

        #Verifica se altura e nem a largura eh nula     
        if (altura ==0 or largura ==0):
            return False

        perimetro = 2*altura + 2*largura
        r = largura/altura                  #aspect do caracter real

        #Verifica se o retangulo do contorno se enquadra com o tamanho do carater
        #de uma placa veicular
        if ((r<rmin or r>rmax)or (perimetro<minPerimetro or perimetro>maxPerimetro)):
            return False
        else:
            return True

    def removerChars(self,text):
        str = "[']!@#%¨&*()_+:;><^^}{`?|~¬\/=,.'´ºª»- "
        for x in str:
            text = text.replace(x,"")
            
        return text  

    def segmentaCaracter(self):
        # aumenta a resoluca da imagem da placa em 4x
        img = cv.resize(self.placa, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
        
        #Converte a imagem para escala de cinza
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #Filtra a imagem pela mediana
        median = cv.medianBlur(img_gray,5)
        
        # Binariza a imagem (preto e branco) por Binarizacao de Otsu
        ret, threshold = cv.threshold(median, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        if self.showsteps:
            cv.imshow("Limiar", threshold)

        img2 = img.copy()

        #Operações morfológicas para remover ruidos
        kernel = np.ones((2,2),np.uint8)
        dilation = cv.dilate(threshold,kernel,iterations=2)
        if self.showsteps:
            cv.imshow('dilata',dilation)

        #Encontra todos os contornos da imagem
        _, contours, hierarchy = cv.findContours(threshold,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        if self.showsteps:
            cv.drawContours(img2, contours, -1, (0,255,0), 1)    #desenha tods os contornos em verde
            cv.imshow('ContornosPlaca',img2)
        
        #Cria uma mascara todas branca
        frase = np.zeros((img.shape[0],img.shape[1]+70),np.uint8)
        frase[::] = 255        #Preenche a imagem de branco

        #Variaveis para ser utilizada abaixo
        caracterList =[]    #Lista de caracteres detectados
        x1 = 5
        y1 = 30

        #Ordena os contornos encontrados da esquerda para direita
        sorted_Contours = sorted(contours,key = self.x_cord_contour,reverse = False)

        for contour in sorted_Contours:
            perimeter = cv.arcLength(contour,True)

            #Calcula um retangulo em volta dos contornos
            (x, y, lar, alt) = cv.boundingRect(contour)
            
            #Verifica se os retangulos obtidos acima se enquadram no tamanho de um caracter da placa
            if(self.verificaChar(lar,alt)):
                if self.showsteps:
                    cv.rectangle(img, (x, y), (x + lar, y + alt), (255, 0, 0), 2)
                    cv.rectangle(dilation.copy(), (x, y), (x + lar, y + alt), 0, 2)
                    cv.imshow('Caracteres Segmentados Imagem Original',img)                       
                    cv.imshow('Caracteres Segmentados Preto e Branco',dilation)
                
                roiCaracter = dilation[y:y+alt,x:x+lar] #Recorta cada um dos caracter e salva uma lista
                #roiCaracter = threshold[y:y+alt,x:x+lar]
                frase[y1:y1+alt,x1:x1+lar] = roiCaracter
                x1 = x1+lar + 10
                if self.showsteps:
                    cv.imshow('Frase',frase)
                    cv.waitKey(0)
                
        self.imagemFrase =  frase       
        return frase

    def reconhecimentoOCR(self,imagem):
        self.placa = imagem.copy()
        #salva a imagem da placa no atributo da classe
          
        #chama a função para segmentar a placa 
        frase = self.segmentaCaracter()

        # Aplica um desfoque na Imagem
        kernel =5
        gaussiano = cv.GaussianBlur(frase, (kernel, kernel), 0)
        
        # Grava o resultado na raiz do codigo
        cv.imwrite('caracter.jpg', gaussiano)

        # Abre a imagem gravada com o modulo PIL, pois o pytesseract so entende imagem
        #desse modulo para ser feita o reconhecimento OCR
        # e salva o resultado na variavel saida
        imagem = Image.open('caracter.jpg')

        #Chama o modulo pytesseract para o reconhecimento dos caracteres da frase segmentada
        saida = tesseract.image_to_string(imagem, lang='eng')
        self.textPlate = saida
        return saida

     




