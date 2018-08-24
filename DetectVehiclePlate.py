try:
    import Image
except ImportError:
    from PIL import Image

import numpy as np
import cv2 as cv
import os
import math

class DetectVehiclePlate:
    inputImage=[]
    placa=[]
    showsteps = False
    placaAltura=0
    placaLargura=0

    #Construtor
    def __init__(self,img_path,steps):
        self.showsteps = steps
        self.inputImage = cv.imread(img_path)
        self.placaAltura = 130
        self.placaLargura = 400
        

    #Metodo para detecção de borda por Canny 
    def canny(self,img, sigma=0.33):
        # Computa a mediana da imagem recebida
        v = np.median(img)
 
        #Aplica o método automático de Canny usando a mediana computada
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv.Canny(img,lower,upper)   # executa o algoritmo de Canny
                                # Canny(source, Limiar_Inferior, Limiar_Superior
        return edges

    
    def verifySizes(self,altura,largura):
        #Método para Verificar se o contorno está dentro do padrão de uma placa
        error =0.3;
        aspect =3.0769 # Placa Spain size: 52x11 aspect 52/11 = 4.7272
                                 # Placa Brazil size 400x130 aspect 3.0769
        perimetroMax = 800
        perimetroMin = 300

        #Min e Max area. Todos os outros tamanhos são discartados
        Min = 30*aspect*30 # area minima
        Max = 130*aspect*130 #area maxima

        #Pega Areas que respeitao o ratio
        rmin = aspect - aspect*error
        rmax = aspect + aspect*error
        
        if (altura ==0 or largura ==0):
            return False

        perimetro = 2*altura + 2*largura
        area = altura * largura    #altura*largura
        r = largura/altura
        
        if ((area<Min or area>Max)or(r<rmin or r>rmax)or (perimetro < perimetroMin or perimetro>perimetroMax)):
            return False
        else:
            print(r)
            return True

    def possiveisPlacas(self,img_bordas):
        #Achando os contornos das possíveis placas
        #recebe as bordas da imagem original
        _, contours, hierarchy = cv.findContours(img_bordas,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        if(self.showsteps):
            img  = np.zeros((img_bordas.shape[0],img_bordas.shape[1],3),np.uint8)
            img = self.inputImage.copy()
            cv.drawContours(img, contours, -1, (0,255,0), 3)    #desenha todos os contornos em verde
            cv.imshow('contornos',img)

        #Criar uma funcao finding place
        i=0
        lista=[]
        img1 = self.inputImage.copy()
        for contour in contours:
            #Verifica se é um contorno fechado
            perimeter = cv.arcLength(contour,True)
            
            #aproxima os contornos da forma correspondente
            approx = cv.approxPolyDP(contour,0.03*perimeter,True)
            
            #verifica se é um quadrado ou retângulo de acordo com os vertices encontrados 
            if (len(approx) >1 and len(approx)<=10):    

                if (self.showsteps):
                    #Contorna a imagem em azul
                    cv.drawContours(img,[contour],-1,(255,0,0),3)
                    cv.imshow('contornos2',img)

                #Calcula um retângulo aproximado em volta do contorno
                #(x,y)-> Coordenada 0,0 do retangulo na imagem
                #(lar,alt) -> largura e altura do retângulo
                (x, y, lar, alt) = cv.boundingRect(contour)
              
                if self.verifySizes(alt,lar):                    #verifica se o contorno bate com o tamanho de uma placa
                    (x, y, lar, alt) = cv.boundingRect(contour)
                    
                    #desenha os retângulos na imagem orignal de possíveis placas
                    if self.showsteps:
                        cv.rectangle(img1, (x, y), (x + lar, y + alt), (0, 0, 255), 2)
                        cv.imshow('Possiveis Placas',img1)
                                                                   
                    #adiciona as possíveis placas em uma lista.
                    lista.append(contour)
                    i = i+1
        print(str(i) + ' Possiveis Placas')
        return lista
        

    def descide(self,lista,xo,yo):
        ##Recebe a lista com os contornos e a coordenada do centro da imagem
        #Essa função escolhe qual é o contorno que tem mais chance de ser a placa
        #O contorno que estiver mais ou centro da imagem será considerado placa
        minDist = 100000000    #infinito
        placa=[]

        #Faz a iteração para cada contorno da lista
        for n in lista:
            #Calcula o um retangulo aproximado do contorno
            (x, y, lar, alt) = cv.boundingRect(n)

            #Calcula a distância entre as coordenadas desejadas
            dist = math.sqrt((((x+lar/2)- xo)**2) +(((y+alt/2) - yo)**2))
            if(dist< minDist):
                minDist = dist
                contorno = n
                xmin = x
                ymin = y
                larmin = lar
                altmin = alt
        return xmin,ymin,larmin,altmin,minDist,contorno


    def detectaPlaca(self):

        img = self.inputImage.copy()               
        ##Conversaçao da imagem para tons de cinza
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        ##Filtro pela mediana para tirar ruído sal e pimenta
        median = cv.medianBlur(gray,5)

        ##Chama função canny para achar as bordas da imagem
        canny = self.canny(median)
        if(self.showsteps):
            cv.imshow('gray',gray)
            cv.imshow('Mediana',median)
            cv.imshow('canny',canny)
        
        ##Chama função para detectar possiveis placas
        lista = self.possiveisPlacas(canny)

        ##verifica se na lista nao esta vazia
        if lista:
            #Caso haja mais de um placa detectada, o algoritimo chama a função "decide" para selecionar a
            #a placa mais ou centro da imagem
            x,y,lar,alt,dist,contorno = self.descide(lista,self.inputImage.shape[1]/2,self.inputImage.shape[0]/2)

            #Secciona a imagem [ROI]
            roi = self.inputImage[y-5:y+alt+5,x-5:x+lar+5].copy()
            cv.imshow('roi',roi)
            cv.imwrite('roi.jpg',roi)       #Guarda a imagem da placa

            #Desenha a localizacao da placa na imagem original
            cv.rectangle(self.inputImage, (x, y), (x + lar, y + alt), (0, 0, 255), 2)
            cv.imshow('Placa',self.inputImage)

            self.placa = roi.copy()
            return self.placa
            
        else:
            print('Placa não detectada')
            return False


                
