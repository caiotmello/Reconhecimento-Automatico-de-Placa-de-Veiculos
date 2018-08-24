import numpy as np
import cv2 as cv
import os
import math
import pytesseract  as ocr
import tkinter
from PIL import Image
from matplotlib import pyplot as plt

##Métode de Canny para detecção de bordas
def canny(img, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(img)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv.Canny(img,lower,upper)   # executa o algoritmo de Canny
                                # Canny(source, Limiar_Inferior, Limiar_Superior
    return edges

##Função VerifySizes

#Verifica se RotatedRect está dentro do padrão de uma placa
def verifySizes(altura,largura):
    error =0.3;
    aspect =4.7272 # Placa Spain size: 52x11 aspect 52/11 = 4.7272
                             # Placa Brazil size 40x13 aspect 3.0769
    perimetroMax = 700
    perimetroMin = 300

    #Min e Max area. Todos os outros tamanhos são discartados
    Min = 30*aspect*30 # area minima
    Max = 80*aspect*80 #area maxima

    #Pega Areas que respeitao o ratio
    rmin = aspect - aspect*error
    rmax = aspect + aspect*error
    
    if (altura ==0 or largura ==0):
        return False

    perimetro = 2*altura + 2*largura
    area = altura * largura    #altura*largura
    r = largura/altura
    #if r < 1:               # verifica se a divisao da altura pela largura     
    #    r = altura/largura  #não é menor que 1
    if ((area<Min or area>Max)or(r<rmin or r>rmax)or (perimetro < perimetroMin or perimetro>perimetroMax)):
        return False
    else:
        return True

##Maximaxe Contrast
#
def maximizeContrast(imgGray):
    altura = imgGray.shape[0]
    largura = imgGray.shape[1]

    imgTopHat = np.zeros((altura,largura,1),np.uint8)
    imgTopHat = np.zeros((altura,largura,1),np.uint8)

    structurinElement = cv.getStructuringElement(cv.MORPH_RECT,(3,3))

    imgTopHat = cv.morphologyEx(imgGray,cv.MORPH_TOPHAT,structurinElement)
    imgBlackHat = cv.morphologyEx(imgGray,cv.MORPH_BLACKHAT,structurinElement)

    imgGrayPlusTopHat = cv.add(imgGray,imgTopHat)
    imgGrayPlusTopHatMinusBlackHat = cv.subtract(imgGrayPlusTopHat,imgBlackHat)

    return imgGrayPlusTopHatMinusBlackHat


##Recebe a lista com os contornos e a coordenada do centro da imagem
#Essa função escolhe qual é o contorno que tem mais chance de ser a placa
#O contorno que estiver mais ou centro da imagem será considerado placa
def descide(lista,xo,yo):
    minDist = 100000000    #infinito
    placa=[]
    for n in lista:
        (x, y, lar, alt) = cv.boundingRect(n)

        #Calcula a distância entre as coordenadas desejadas
        dist = math.sqrt((((x+lar/2)- xo)**2) +(((y+alt/2) - yo)**2))
        print(dist)
        if(dist< minDist):
            minDist = dist
            contorno = n
            xmin = x
            ymin = y
            larmin = lar
            altmin = alt
    return xmin,ymin,larmin,altmin,minDist,contorno

##Função para fazer o reconhecimento das letras da placa       
def reconhecimentoOCR(imagem_path):
    pathSaida ='./'
    #Faz a leitura da imagem
    img = cv.imread(imagem_path)

    #img_limpa = removeExcessos(img)
    #listaCaracter = segmentaCaracter(img)
    frase = segmentaCaracter(img)

    #cv.imshow('caracte',caracter)
    # Aplica um desfoque na Imagem
    kernel =5
    gaussiano = cv.GaussianBlur(frase, (kernel, kernel), 0)
    #cv.imshow("Desfoque", img)

    # Grava o resultado na raiz do codigo
    cv.imwrite(pathSaida +'caracter.jpg', gaussiano)

    # Abre a imagem gravada com o modulo PIL, pois o pytesseract so entende imagem
    #desse modulo para ser feita o reconhecimento OCR
    # e salva o resultado na variavel saida
    imagem = Image.open(pathSaida + 'caracter.jpg')
    saida = ocr.image_to_string(imagem, lang='eng')
    print(saida)
 
    # Realiza um filtro nos caracteres obtidos 
    # eliminando possiveis ruidos reconhecidos
    if len(saida) > 0:
        print(saida)
        texto = removerChars(saida)
    else:
        texto = "Reconhecimento Falho"
 
    janela = tkinter.Tk()
    tkinter.Label(janela, text=texto, font=("Helvetica", 50)).pack()
    janela.mainloop()


def removerChars(text):
    str = "!@#%¨&*()_+:;><^^}{`?|~¬\/=,.'ºª»-"
    for x in str:
        text = text.replace(x, '')
    return text  

def removeExcessos(roi):
    
    # aumenta a resoluca da imagem da placa em 4x
    img = cv.resize(roi, None, fx=4, fy=4, interpolation=cv.INTER_CUBIC)
    
    #Converte a imagem para escala de cinza
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    median = cv.medianBlur(img_gray,5)
    # Binariza a imagem (preto e branco)
    ret, threshold = cv.threshold(median, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    #ret, threshold = cv.threshold(median, 40, 255, cv.THRESH_BINARY)
    cv.imshow("Limiar", threshold)

    img2 = img.copy()
    
    im2, contours, hierarchy = cv.findContours(threshold,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, contours, -1, (0,255,0), 3)    #desenha tods os contornos em verde
    cv.imshow('placasad',img)

    #Operações morfológicas para remover ruidos
    kernel = np.ones((3,3),np.uint8)
    dilation = cv.dilate(threshold,kernel,iterations=2)
    median = cv.medianBlur(dilation,5)
    erosion = cv.erode(dilation,kernel,iterations=2)
    cv.imshow('dilata',dilation)
    cv.imshow('erode',erosion)
    cv.imshow('median',median)

    #Cria uma mascara todas branca
    mask = np.zeros((img.shape[0],img.shape[1]),np.uint8)
    
    for contour in contours:
        perimeter = cv.arcLength(contour,True)
        #approx = cv.approxPolyDP(contour,0.04*perimeter,True)
        #if(perimeter> 150 and len(approx>3)):
        if(perimeter> 300):
            approx = cv.approxPolyDP(contour,0.04*perimeter,True)
            #cv.drawContours(img2,[approx],-1,(0,0,255),3)
            cv.drawContours(mask,[contour],0,255,-1)         #Cria uma mascara a partir do contor 
            #cv.imshow('approx',img2)                        #da placa
            cv.imshow('mask',mask)
            print(perimeter)

            # Combine the two images to get the foreground.
            mask_inv = cv.bitwise_not(mask)
            im_out = mask_inv | erosion
            cv.imshow('final',im_out)

    #caio =np.zeros((img.shape[0],img.shape[1]),np.uint8)
    #caio = im_out.copy()
    #im2, contours, hierarchy = cv.findContours(im_out,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    #cv.drawContours(caio, [contours],0, 255, 0)    #desenha tods os contornos em verde
    #cv.imshow('Ultima',caio)
    
    return im_out

def segmentaCaracter(img_in):
    # aumenta a resoluca da imagem da placa em 4x
    img = cv.resize(img_in, None, fx=4, fy=4, interpolation=cv.INTER_CUBIC)
    
    #Converte a imagem para escala de cinza
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    median = cv.medianBlur(img_gray,5)
    # Binariza a imagem (preto e branco)
    ret, threshold = cv.threshold(median, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    #ret, threshold = cv.threshold(median, 40, 255, cv.THRESH_BINARY)
    cv.imshow("Limiar", threshold)

    img2 = img.copy()
    
    im2, contours, hierarchy = cv.findContours(threshold,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, contours, -1, (0,255,0), 1)    #desenha tods os contornos em verde
    cv.imshow('ContornosPlaca',img)

    #Operações morfológicas para remover ruidos
    kernel = np.ones((3,3),np.uint8)
    dilation = cv.dilate(threshold,kernel,iterations=2)
    median = cv.medianBlur(dilation,5)
    erosion = cv.erode(dilation,kernel,iterations=2)
    #cv.imshow('dilata',dilation)
    #cv.imshow('erode',erosion)
    #cv.imshow('median',median)

    #Cria uma mascara todas branca
    phrase = np.zeros((img.shape[0],img.shape[1]),np.uint8)
    phrase[::] = 255        #Preenche a imagem de branco
    caracterList =[]
    x1 = 30
    y1 = 30

    sorted_Contours = sorted(contours,key = x_cord_contour,reverse = False)
    for contour in sorted_Contours:
        perimeter = cv.arcLength(contour,True)
        #approx = cv.approxPolyDP(contour,0.04*perimeter,True)
        #if(perimeter> 150 and len(approx>3)):
        #((x,y),(alt,lar),angle) = cv.minAreaRect(contour)
        (x, y, lar, alt) = cv.boundingRect(contour)
        if(verificaChar(lar,alt)):
            cv.rectangle(img, (x, y), (x + lar, y + alt), (255, 0, 0), 2)
            cv.rectangle(erosion.copy(), (x, y), (x + lar, y + alt), 0, 2)
            cv.imshow('approx',img)                       
            cv.imshow('mask',erosion)
            roiCaracter = dilation[y:y+alt,x:x+lar] #Recorta cada um dos caracter e salva uma lista
            cv.imshow('Caracter',roiCaracter)
            phrase[y1:y1+alt,x1:x1+lar] = roiCaracter
            x1 = x1+lar +10
            #print(perimeter)
            cv.imshow('Frase',phrase)
            cv.waitKey(0)
              
    return phrase

#retorna a coordenada X para o contorno 
def x_cord_contour(contours):
    M = cv.moments(contours)
    return (int(M['m10']/M['m00']))
    
def verificaChar(largura,altura):
    error =0.3;
    aspect =0.61290 # Placa Brasil 38cmx62cm aspect 38/62 = 0.61290
                             # Placa Europa size 45.0f/77.0f aspect 3.0769

    #Min e Max area. Todos os outros tamanhos são discartados
    minPerimetro = 100 # area min
    maxPerimetro = 600#area maxima

    #Pega Areas que respeitao o ratio
    rmin = 0.2         #O minimo é do caracter I e 1 qu etem dimensao 13x62
    rmax = aspect + aspect*error

    #If an area is higher than 80 percent, we consider that region to be a black
    #block, and not a character
    #cv2.countNonZero(src)
    
    if (altura ==0 or largura ==0):
        return False

    perimetro = 2*altura + 2*largura
    area = altura * largura    #altura*largura
    r = largura/altura
    if ((r<rmin or r>rmax)or (perimetro<minPerimetro or perimetro>maxPerimetro)):
        return False
    else:
        return True
    
############## Main Loop ##################
path = './'
showsteps = 1
img = cv.imread('./Imagens/Foto/propertisVideo.jpg')
imagem_original = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
imagem_original = img.copy()

##Conversaçao da imagem para tons de cinza
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#contraste = maximizeContrast(gray)
#cv.imshow('contraste',contraste)

##Filtro pela mediana para tirar ruído sal e pimenta
median = cv.medianBlur(gray,5)

##Chama função canny para achar as bordas da imagem
canny = canny(median)
if(showsteps):
    cv.imshow('gray',gray)
   # cv.imshow('Mediana',median)
    cv.imshow('canny',canny)

#kernel = np.ones((3,3),np.uint8)
#dilation = cv.dilate(threshold,kernel,iterations=2)
#dilation = cv.dilate(canny,kernel,iterations=1)
#erosion = cv.erode(dilation,kernel,iterations=1)
#cv.imshow('erosion',erosion)

#Achando os contornos das possíveis placas
#im2, contours, hierarchy = cv.findContours(erosion,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
im2, contours, hierarchy = cv.findContours(canny,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
cv.drawContours(img, contours, -1, (0,255,0), 3)    #desenha tods os contornos em verde
if(showsteps):
    cv.imshow('contornos',img)

#Criar uma funcao finding place
i=0
lista=[]
for contour in contours:
    #Verifica se é um contorno fechado
    perimeter = cv.arcLength(contour,True)
    #if perimeter >1 and perimeter <2000:
    #aproxima os contornos da forma correspondente
    #hull = cv.convexHull(contour)
    approx = cv.approxPolyDP(contour,0.03*perimeter,True)
    #verifica se é um quadrado ou retângulo de acordo com os vertices encontrados
    #if (len(hull) >10 and len(hull)<=30):
    if (len(approx) >1 and len(approx)<=10):    
        #Contorna a imagem
        cv.drawContours(img,[approx],-1,(255,0,0),3)
        cv.imshow('contornos',img)
        #print(verifySizes(cv.minAreaRect(contour)))
        #((x,y),(lar,alt),angle) =cv.minAreaRect(contour)
        (x, y, lar, alt) = cv.boundingRect(contour)
        if verifySizes(alt,lar):                    #verifica se o contorno bate com o de uma placa
            (x, y, lar, alt) = cv.boundingRect(contour)
            cv.rectangle(img, (x, y), (x + lar, y + alt), (0, 0, 255), 2)
            cv.imshow('Possiveis Placas',img)
           # #print((x, y, lar, alt))
           # print(len(approx),len(hull),2*lar+2*alt)
                                    
            #adiciona as possíveis placas em uma lista.
            lista.append(contour)
            i = i+1


print(str(i) + ' Possiveis Placas')

#Verifica se a lista não está vazia
if lista:
    #Caso haja mais de um placa detectada, o algoritimo chama a função "decide" para selecionar a
    #a placa mais ou centro da imagem
    x,y,lar,alt,dist,contorno = descide(lista,imagem_original.shape[1]/2,imagem_original.shape[0]/2)

    #Secciona a imagem [ROI]
    roi = imagem_original[y-5:y+alt+5,x-5:x+lar+5].copy()
    cv.imshow('roi',roi)
    cv.imwrite('roi.jpg',roi)

    #desenha a localizacao da placa na imagem original
    cv.rectangle(imagem_original, (x, y), (x + lar, y + alt), (0, 0, 255), 2)
    cv.imshow('Placa',imagem_original)

    #chama o algoritmo para detectar os caracters da placa
    reconhecimentoOCR(path +'roi.jpg')

    #teste(roi)
else:
    print('Placa não detectada')
#cv.line(imagem_original, ((imagem_original.shape[1]/2),0), ((imagem_original.shape[1]/2), imagem_original.shape[0]), (0, 0, 255), 1)
#cv.line(imagem_original, (0, (imagem_original.shape[0]/2)), (imagem_original.shape[1], (imagem_original.shape[0]/2)), (0, 0, 255), 1)

#phare = ocr.image_to_string(Image.open('roi.jpg'),lang='por')
#print(phare)

