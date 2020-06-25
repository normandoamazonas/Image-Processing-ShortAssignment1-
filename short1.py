'''Normando de Campos Amazonas Filho, 11561949
Image Processing, SCC0251_Turma01_1Sem_2020_ET
Short Assignment 1: Filtering in Fourier Domain 
https://github.com/normandoamazonas/ShortAssignment1'''

import numpy as np
import imageio as imageio
import cmath
'''Parameters input'''   
filename = str(input()).rstrip()
input_img = imageio.imread(filename)
T = float(input()) # 0<= T <=1 threshold

#DFT based on 1D DFT from Moacir 
def DFT(image):
    n,m = image.shape #getting the dimensions of the image
    F = np.zeros(image.shape,dtype = np.complex64) #creating the image that is going to store the transformation
    for u in np.arange(n): #it´s going to run the row of the transformation
        for v in np.arange(m): #it´s going to run the column of the transformation
            #soma = 0.0 #sum value to help the understanding
            for x in np.arange(n): #it´s going to run the row of the original
                for y in np.arange(m): #it´s going to run the column of the original
                    F[u,v]+=(image[x,y]*np.exp(-1j *2*np.pi*(float(u*x) /n + float(v*y)/m)))/np.sqrt(n*m)
                    #value = image[x,y] 
                    #e = np.exp(-1j *2*np.pi*(float(u*x) /n + float(v*y)/m)) #computing the transformation
                    #soma+=value * e
            #F[u,v] = soma/np.sqrt(n*m)
    return F  

#IDFTbased on DFT
def D(transformada,shape):
    n,m = shape
    inverse=np.zeros(shape)
    for x in np.arange(n): #iterate the lines
        for y in np.arange(m): #iterate the columns
            soma=0.0 
            for u in np.arange(n): #the first sumatory
                for v in np.arange(m): # the second sumatory
                    inverse[x,y]+=(image[x,y]*np.exp(-1j *2*np.pi*(float(u*x) /n + float(v*y)/m)))/np.sqrt(n*m)
                    value = transformada[u, v]
                    e = np.exp(1j * 2*np.pi* (float(u*x)/ n + float(v*y)/m)) # computing the inverse
                    soma +=value*e
            inverse[x, y] = (np.abs(soma)/np.sqrt(n*m))
    return inverse
def IDFT(transformada,shape):
    n,m = shape
    inverse=np.zeros(shape)
    for x in np.arange(n): #iterate the lines
        for y in np.arange(m): #iterate the columns
            soma=0.0 
            for u in np.arange(n): #the first sumatory
                for v in np.arange(m): # the second sumatory
                    soma += transformada[u, v] * np.exp(1j * 2*np.pi* (float(u*x)/ n + float(v*y)/m))
                    #value = transformada[u, v]
                    #e = np.exp(1j * 2*np.pi* (float(u*x)/ n + float(v*y)/m)) # computing the inverse
                    #soma +=value*e
            inverse[x, y] = np.abs(soma)/np.sqrt(n*m)
            #inverse[x, y] = (np.abs(soma)/np.sqrt(n*m))
    return inverse

transformado = DFT(input_img) #computing the DTF transform
transformado_temp = np.abs(transformado) #computing the abs, real with imaginary

#BEGIN - this operation is to calculate the second peak
flat=transformado_temp.flatten()
flat.sort()
#second peak
p2 = flat[-2]
#END - this operation is to calculate the second peak


#theshold p2*T 
Threshold = p2*T
#mask |F|< p2*T
mask_data = transformado_temp < Threshold #compute the values under  the Threshold
F_Coeff = sum(sum(mask_data)) # calculate how many values are under  the Threshold
#filtered
transformado[mask_data] = 0.0 #applying |F|< p2*T

#computing the inverse DFT
InverseImage = IDFT(transformado,input_img.shape)


'''
Threshold=%.4f
Filtered Coefficients=%d
Original Mean=%.2f
New Mean=%.2f
'''
#printing the required values
print("Threshold=%.4f"%Threshold)
print("Filtered Coefficients=",F_Coeff)
print("Original Mean=%.2f"%np.mean(input_img))
print("New Mean=%.2f"%np.mean(InverseImage))
