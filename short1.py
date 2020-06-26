'''Normando de Campos Amazonas Filho, 11561949
Image Processing, SCC0251_Turma01_1Sem_2020_ET
Short Assignment 1: Filtering in Fourier Domain 
https://github.com/normandoamazonas/ShortAssignment1'''

import numpy as np
import imageio as imageio
import cmath
import time 
'''Parameters input'''   
filename = str(input()).rstrip()
input_img = imageio.imread(filename)
T = float(input()) # 0<= T <=1 threshold

#DFT based on 1D DFT from Moacir 
def DFT(image): 
    n,m = image.shape #getting the dimensions of the image
    F = np.zeros(image.shape,dtype = np.complex64) #creating the image that is going to store the transformation
    x = np.arange(n) 
    for u in np.arange(n): #it´s going to run the row of the transformation
        for v in np.arange(m): #it´s going to run the column of the transformation
            for y in np.arange(m): #it´s going to run the column of the original
                e = np.exp( (-1j*2*np.pi) * (((u*x)/n)+((v*y)/m)) ) #computing the transformation
                soma=np.sum(image[:,y]*e)
                F[u,v] += soma/np.sqrt(n*m)
    return F 

def IDFT(transformada,shape):
    n,m = shape
    inverse=np.zeros(shape,dtype = np.complex64)
    u = np.arange(n)
    for x in np.arange(n): #iterate the lines
        for y in np.arange(m): #iterate the columns
            for v in np.arange(m): # the second sumatory
                e= np.exp( (1j*2*np.pi) * (((u*x)/n)+((v*y)/m)) ) # computing the inverse
                soma = np.sum(transformada[:, v] * e)
                inverse[x, y]+= soma/np.sqrt(n*m)
    return np.abs(inverse)

#start = time.time()
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

#end = time.time()
#elapsed = end - start
#print("Running time: %.5f sec." %  elapsed)

'''
Threshold=%.4f
Filtered Coefficients=%d
Original Mean=%.2f
New Mean=%.2f
'''
#printing the required values
print("Threshold=%.4f"%Threshold)
print("Filtered Coefficients=%d"%F_Coeff)
print("Original Mean=%.2f"%np.mean(input_img))
print("New Mean=%.2f"%np.mean(InverseImage))
