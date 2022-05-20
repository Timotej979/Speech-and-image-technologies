import numpy as np
import cv2
import matplotlib.pyplot as plt
from functools import reduce
import os

def skin_color_filter(im):
    # vhod: barvna slika
    # izhod: odziv filtra za barvo kože

    # preberemo sliko po kanalih
    red, green, blue = [im[:, :, x] for x in range(3)]
    
    #generacija maske ter dodatni pogoji
    mask = reduce(np.logical_and, (red > 95, green > 40, blue > 20, (reduce(np.maximum, (red, green, blue)) - reduce(np.minimum, (red, green, blue))) > 15, np.abs(red - green) > 15,red > green, red > blue))
    
    #pretvorba maske v uint8
    mask = mask.astype("uint8")   
    return mask

def fill(mask):

    # kopiramo vhodno masko
    image_floodfill = mask.copy() 
    
    # preberemo velikost maske
    height, width = mask.shape[:2]
    
    # priprava 2x večje maske za novo sliko
    new_mask = np.zeros((height + 2, width + 2), np.uint8)

    # zapolnimo slike z vrednostmi 255
    cv2.floodFill(image_floodfill, new_mask, (0,0), 255)
    
    # naredimo inverz slike
    image_floodfill_inv = cv2.bitwise_not(image_floodfill)
    
    # izhod je bitni ALI polne maske in inverza slike
    image_out = mask | image_floodfill_inv    
    return image_out


def dilate(mask):
    # vhod: maska
    # izhod: maska, očiščena z dilatacijo s 5*5 kvadratnim strukturnim elementom
    
    # strukturni element 5x5
    #strutural_element_5 = np.ones((5,5),np.uint8)
    
    # isti element z vgrajenim fjem
    element_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5), (-1,-1) )
    
    # kopiramo masko
    im_dilate = mask.copy()
    
    # preberemo velikost maske
    height, width = mask.shape[:2]
    
    # pripravimo novo masko
    new_mask = np.zeros((height, width), np.uint8)
    
    # uporaba ukaza za izračun dilatacije
    cv2.dilate(im_dilate, element_5, new_mask, iterations=1 )
    
    image_out = new_mask 
    return image_out


def clean_skin_mask(mask):
    # kombinacija erozije in dilatacije
    
    # kopiramo masko
    image_cleanskin = mask.copy() 
    
    # zapolnimo novo masko z prejšnjo
    image_cleanskin = fill(image_cleanskin) 
    
    # združimo dilatacijo in erozijo v eno masko
    image_out = dilate(image_cleanskin)
    return image_out
    

def rgb2gray(im):
    # vhod: barvna slika
    # izhod: sivinska slika, dobljena s povprečenjem kanalov
    
    # razširimo sliko iz uint8 v uint16
    image_float = im.astype("uint16")
    
    # preberemo sliko po kanalih
    red, green, blue = [image_float[:, :, x] for x in range(3)]
    
    # pretvorba RGB vrednosti kanalov v sive vrednosti
    gray_value = (red + green + blue) / 3
    
    # pretvorba float v int
    gray_value = gray_value.astype("uint8")
    return gray_value


def smooth_gray_image(im):
    # vhod: sivinska slika
    # izhod: sivinska slika, zglajena s 7*7 Gaussovim jedrom s sigma=2
    
    # kopiramo masko
    image_smooth = im.copy() 

    # uporabimo gaussov filter
    cv2.GaussianBlur(im, (7,7),2,image_smooth, borderType=cv2.BORDER_DEFAULT)
    return image_smooth 


def gradient_approximation(im):
    # vhod: zglajena siva slika
    # izhod: približek amplitude gradientov (v float32)
    
    # kopiramo masko
    image_sobel = im.copy()
    
    # masko spremenimo iz int v float
    image_sobel = image_sobel.astype("float32")
    
    # pripravimo Sobelov operator/matriko/element za izračun v X in Y smereh
    sobelX = np.array([[-1, 0, 1], [-2,0,2], [-1,0,1]], dtype="float32")
    sobelY = np.transpose(sobelX)
    
    # filtriramo masko po X in Y smeri
    image_x = cv2.filter2D(image_sobel, cv2.CV_32F, sobelX)
    image_y = cv2.filter2D(image_sobel, cv2.CV_32F, sobelY)
    
    # seštevek komponent po X in Y smeri po pitagorovem izreku
    image_border = np.sqrt(np.square(image_x) + np.square(image_y))   
    return image_border


def mask_edges(edges, mask):
    # vhoda: amplituda gradientov, maska barve kože
    # izhod: slika amplitude gradientov, maskirana na področja ki ustrezajo odzivu filtra barve kože

    # robove pomnožimo z očiščeno masko
    image_maskededge = np.multiply(edges, mask)
    
    # skaliramo vrednosti v območje 0-255
    image_maskededge = image_maskededge * (255 / np.amax(image_maskededge))
    
    # določimo mejo 40 ter vrnemo primeren bool
    image_out_bool = (image_maskededge > 40)
    
    # spremenimo vrednosti bool v uint8
    image_out = image_out_bool.astype("uint8")
    return image_out


def threshold_edges(edges):
    # vhod: maskirana slika amplitude gradientov
    # izhod: binarna slika, rezultat upragovljanja slike amplitude gradientov s pragom 40
    
    # kopiramo masko
    image_maskededge = edges.copy()
    
    # skaliramo vrednosti v območje 0-255
    image_maskededge = image_maskededge*(255/np.amax(im_maskededge))
    
    # določimo mejo 40 ter vrnemo primeren bool
    im_out_bin = (im_maskededge > 40)
    
    # združeno z funkcijo mask_edges
    im_out = im_out_bin*255
    im_out = im_out.astype("uint8")
    return im_out


def fit_ellipse(bin_edges):
    # vhod: binarna slika robov, maskiranih s področjem kože
    # izhod: parametri elipse
    
    # tabela za točke
    elipse_points = []
    
    # zanke, ki tečejo čez masko
    for x in range(bin_edges.shape[1]):
        for y in range(bin_edges.shape[0]):
            if bin_edges[y, x] > 0:
                elipse_points.append((x, y))
    
    # pretvorba tabele v float32
    elipse_points = np.array(elipse_points).astype("float32")
    
    # določitev eliptične maske
    ellipse_mask = cv2.fitEllipse(elipse_points)
    return ellipse_mask


def draw_ellipse(im, ell):
    # vhod: barvna slika in parametri elipse
    # izhod: slika z vrisano elipso
    
    # pridobitev podatkov iz strukture
    center, axes, angle = ell
    
    # pretvorba float kordinat v int 
    center = (int( round(center[0])), int(round(center[1])) )
    axes = (int( round(axes[0]/2)), int(round(axes[1]/2)) )
    angle = int( round(angle) )
    
    # izpis kordinat
    print(center, axes, angle)

    # pretvorba vhodne slike v float64
    image_in = im.astype("float64")
    
    # izvedba risanja eliptične maske na sliko
    face_det = cv2.ellipse(image_in, center=center, axes=axes, angle=angle, startAngle=0.0, endAngle=360.0, color=(255, 0, 0), thickness=2)
                           
    # pretvorba slike iz float64 nazaj v uint8
    return face_det.round().astype("uint8")


def prepoznava_obraza(im):
    # vhod: barvna slika 
    # izhod: barvna slika z vrisano elipso
    
    # izvedba vseh funkcij zaporedno 
    image_skin = skin_color_filter(im)
    image_clean = clean_skin_mask(image_skin)

    image_gray = rgb2gray(im)
    image_graysmooth = smooth_gray_image(image_gray)

    image_gradsobel = gradient_approximation(image_graysmooth)
    image_maskededge = mask_edges(image_gradsobel, image_clean)

    elipse_parameter = fit_ellipse(image_maskededge)
    image_with_elipse = draw_ellipse(im, elipse_parameter)
    
    return image_with_elipse


if __name__ == "__main__":
       
    slike = os.listdir("slike")
    
    for slika in slike:
        
        # OSNOVNA SLIKA
        img = cv2.imread(os.path.join("slike",slika))[:, :, ::-1]    
        plt.imshow(img)
        plt.show()
    	
        # ČRNO-BEL OBRAZA
        imgskin = skin_color_filter(img)
        plt.imshow(imgskin, cmap="gray")
        plt.show()
    
        # EROZIJA OBRAZA
        imgclean = clean_skin_mask(imgskin)
        plt.imshow(imgclean, cmap="gray")
        plt.show()
    
        # ČRNO-BELA SLIKA
        imggray = rgb2gray(img)
        plt.imshow(imggray, cmap="gray")
        plt.show()
    	
        # GLAJENJE ČRNO-BELE SLIKE
        imggraysmooth = smooth_gray_image(imggray)
        plt.imshow(imggraysmooth, cmap="gray")
        plt.show()
    	
        # DETEKCIJA ROBOV OBRAZA
        imggradsobel = gradient_approximation(imggraysmooth)
        plt.imshow(imggradsobel, cmap="gray")
        plt.show()
    	
        # EROZIJA ROBOV OBRAZA
        imgmaskededge = mask_edges(imggradsobel, imgclean)
        plt.imshow(imgmaskededge, cmap="gray")
        plt.show()
    	
        # DETEKCIJA OBRAZA
        face = prepoznava_obraza(img) 
        plt.imshow(face)
        plt.show() 

    pass
