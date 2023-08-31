import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QStackedWidget, QComboBox, QDateEdit, QTextEdit,QLineEdit, QFormLayout, QPushButton
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets, QtCore
import skimage  # Importación faltante
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph

from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt5.QtWidgets import QGraphicsScene
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import cm

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2
from scipy import signal
import math
from skimage.transform import radon,rescale,iradon,rotate,resize,iradon_sart
import warnings
warnings.simplefilter("ignore")
from scipy.ndimage import rotate
from skimage import transform as tf
from sklearn.cluster import KMeans

from skimage import color
import pandas as pd
from radiomics import glcm
import SimpleITK as sitk

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score


class InterfazPib(QMainWindow):
    
    def __init__(self):
        super(InterfazPib,self).__init__()
        loadUi('C:/Users/Luciana/Downloads/TPC-Gatti-Guirula-FernandezBettelli.ui', self)  # Carga el archivo .ui creado con Qt Designer
        print("Inicializando la interfaz")
        
        # Conecta los botones a las funciones correspondientes
        self.datos.clicked.connect(self.datos_paciente)
        self.cargarimagen.clicked.connect(self.cargar_imagen)
        self.caracterizar.clicked.connect(self.caracterizar_imagen)
        self.diagnosticar.clicked.connect(self.diagnosticar_imagen)
        self.imagenoriginal.clicked.connect(self.mostrar_imagen)
        self.stacked_widget = self.findChild(QStackedWidget, "stackedWidget")
        # Obtener referencia al QComboBox
        self.combobox = self.findChild(QComboBox, "comboBox")

        # Agregar opciones al QComboBox
        self.combobox.addItems(["Femenino", "Masculino", "No especifica"])

        self.guardar.clicked.connect(self.guardar_pdf)
    def guardar_pdf(self):
        # Obtener los textos de los controles
        n2_text = self.n2.text()
        d2_text = self.d2.text()
        date_text = self.dateEdit.date().toString("dd/MM/yyyy")
        combo_text = self.comboBox.currentText()
        diag_text = self.diagline.text()
        obs_text = self.obs.toPlainText()

         # Abrir el diálogo de guardado de archivo para elegir la ubicación del PDF
        file_dialog = QFileDialog()
        file_dialog.setDefaultSuffix(".pdf")
        file_name=n2_text + "- Diagnostico .pdf"
        file_path, _ = file_dialog.getSaveFileName(self, "Guardar PDF", file_name, "PDF Files (*.pdf)")
        if file_path:
            # Crear un objeto Canvas de ReportLab y generar el PDF
            c = canvas.Canvas(file_path,pagesize=letter)

            # Agregar el título centrado y con un tamaño de fuente más grande
            c.setFont("Helvetica-Bold", 24)
            c.drawCentredString(letter[0] / 2, letter[1] - 50, "Informe oftalmológico")
            # Agregar una línea horizontal debajo del título
            c.setLineWidth(1)
            c.setStrokeColor(colors.black)
            c.line(50, letter[1] - 70, letter[0] - 50, letter[1] - 70)

            c.setFont("Helvetica", 12)

            ## Agregar los demás textos al PDF
            c.setFont("Helvetica", 12)
            c.drawString(50, letter[1] - 120, "Nombre y apellido: " + n2_text)
            c.drawString(50, letter[1] - 140, "Documento de identidad: " + d2_text)
            c.drawString(50, letter[1] - 160, "Fecha de nacimiento: " + date_text)
            c.drawString(50, letter[1] - 180, "Sexo: " + combo_text)
            c.drawString(50, letter[1] - 200, "Diagnóstico: " + diag_text)

            c.drawString(50, letter[1] - 240, "Observaciones: " + obs_text)
            c.drawString(50, letter[1] - 220, self.texto_etiqueta)

            c.drawString(50, letter[1] - 260, "Imágenes y parámetros obtenidos: ")

            c.drawImage('grafs.png', 50, letter[1] - 280-300, width=400, height=300)

            c.drawString(50, letter[1] - 260- 330, "Promedio de parámetros base de datos ")

            c.drawImage('Enfermedad.png', 50, letter[1] - 280-460,width=15*cm,height=4*cm)
            
            # Agregar la imagen al PDF
            # Tomar el archivo CSV y calcular los promedios por clase
            
            # Guardar y cerrar el PDF
            c.save()
    
    
    
    
    def datos_paciente(self):
            # Obtener el índice de la pestaña "datos" en el QStackedWidget
            index = self.stacked_widget.indexOf(self.datos_2)

            # Mostrar la pestaña "im"
            self.stacked_widget.setCurrentIndex(index)

    def cargar_imagen(self):
        # Lógica para cargar una imagen
        print("Botón Cargar Imagen presionado")
        file_path, _ = QFileDialog.getOpenFileName(self, "Cargar imagen", "", "Imágenes (*.png *.jpg *.jpeg)")
        if file_path:
            #Lógica para cargar la imagen utilizando OpenCV
            self.loaded_image = skimage.io.imread(file_path)
            print("Imagen cargada exitosamente: {}".format(file_path))

            self.mostrar_mensaje("Imagen cargada exitosamente: {}".format(file_path))
        else:
            self.mostrar_mensaje("Error al cargar la imagen")
        #self.diagnostico.setText('-')
        
    
        
    def mostrar_imagen(self):
        print("Botón mostrar presionado")
        if hasattr(self, 'loaded_image'):  # Verifica si la imagen ha sido cargada previamente
            print("Shape :", self.loaded_image.shape)

            fig,ax=plt.subplots(2,2)
            f = 18
            #ORIGINAL
            im = ax[0,0].imshow(self.loaded_image, vmin=0, vmax=255)
            ax[0,0].set_title('Imagen original',fontsize=f)
            ax[0, 0].axis('off')  # Ocultar ejes

            #ROJA
            im_r = ax[0,1].imshow(self.loaded_image[:,:,0], vmin=0, vmax=255,cmap='gray')
            ax[0,1].set_title('Imagen roja',fontsize=f)
            ax[0, 1].axis('off')  # Ocultar ejes


            #VERDE
            im_g = ax[1,0].imshow(self.loaded_image[:,:,1], vmin=0, vmax=255,cmap="gray")
            ax[1,0].set_title('Imagen verde',fontsize=f)
            ax[1, 0].axis('off')  # Ocultar ejes
            
            #AZUL
            im_b = ax[1,1].imshow(self.loaded_image[:,:,2], vmin=0, vmax=255,cmap="gray")
            ax[1,1].set_title('Imagen azul',fontsize=f)
            ax[1, 1].axis('off')  # Ocultar ejes
            plt.tight_layout()  # Ajustar los espacios entre los gráficos

            plt.savefig('temp.png')  # Guardar la imagen temporalmente como archivo PNG
            plt.close()  # Cerrar la figura después de guardar la imagen
            

            # Crear objeto QImage a partir del archivo guardado
            qimage = QImage('temp.png')
            pixmap = QPixmap.fromImage(qimage)

            # Acceder al QGraphicsView dentro del QStackedWidget
            stacked_widget = self.findChild(QStackedWidget, 'stackedWidget')

            # Obtener el índice de la pestaña "im" en el QStackedWidget
            index = stacked_widget.indexOf(self.im)

            # Mostrar la pestaña "im"
            stacked_widget.setCurrentIndex(index)
            graphics_view = stacked_widget.widget(index).findChild(QtWidgets.QGraphicsView, 'imagenes')

            # Mostrar la imagen en el QGraphicsView
            graphics_scene = QtWidgets.QGraphicsScene()
            graphics_scene.addPixmap(pixmap)
            graphics_view.setScene(graphics_scene)
            graphics_view.fitInView(graphics_scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

        else:
            self.mostrar_mensaje("No se ha cargado ninguna imagen.")
            


    def mostrar_mensaje(self, mensaje):
        msg_box = QMessageBox()
        msg_box.setText(mensaje)
        palette = QPalette()
        palette.setColor(QPalette.Text, QColor(255, 255, 255))  # Establece el color del texto en blanco (RGB: 255, 255, 255)
        msg_box.setPalette(palette)
        msg_box.exec_()
    
    
    #FUNCIONES PARA CARACTERIZAR
    #FUNCIONES PARA CARACTERIZAR
    #FUNCIONES PARA CARACTERIZAR
    #FUNCIONES PARA CARACTERIZAR
    #FUNCIONES PARA CARACTERIZAR
    #FUNCIONES PARA CARACTERIZAR
    #FUNCIONES PARA CARACTERIZAR
    def eliminar_fondo_circular(self,imagen, center_x, center_y, radius):
        # Crear una imagen en blanco del mismo tamaño que la imagen original
        height, width = imagen.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        # Calcular las coordenadas de los píxeles en la imagen
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        # Calcular la distancia entre los píxeles y el centro del círculo
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)

        # Establecer los píxeles dentro del círculo en la imagen original
        mask[distances <= radius] = imagen[distances <= radius]

        # Establecer los píxeles fuera del círculo en cero
        mask[distances > radius] = 0


        return mask
    
    def gamma_correction(self,image, gamma):
        # Normalizar la imagen en el rango de 0 a 1
        normalized_image = image / 255.0

        # Aplicar la corrección gamma
        corrected_image = np.power(normalized_image, gamma)

        # Escalar los valores nuevamente al rango de 0 a 255
        corrected_image = (corrected_image * 255).astype(np.uint8)

        return corrected_image
    
    def cocu(self,image,coord_x,coord_y,radius):
    #RECIBE IMAGEN GAMMA
        #print("Tipo de image:", type(image))

        image_sitk = sitk.GetImageFromArray(image)  # Convertir NumPy array a SimpleITK image

        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, (coord_x, coord_y), radius, 1, -1)
        #print('1')

        # Aplicar la umbralización de Otsu
        threshold_filter = sitk.OtsuThresholdImageFilter()
        thresholded_image = threshold_filter.Execute(image_sitk)
        #print('2')

        # Convertir la imagen umbralizada a tipo double
        thresholded_image = sitk.Cast(thresholded_image, sitk.sitkFloat64)
        #print('3')
        
        #print("Tipo de image_sitk:", type(image_sitk))
        #print("Tipo de mask:", type(mask))

        
        
        # Convertir la imagen umbralizada a una máscara binaria
        mask_sitk = sitk.GetImageFromArray(mask)
        masked_image = sitk.Mask(thresholded_image, mask_sitk)
        
        #print("Tipo de masked_image:", type(masked_image))

        mask = sitk.GetArrayFromImage(masked_image)

        #print('4')
        
        #print(type(image_sitk))
        #print(mask.shape)
        # Crear una instancia de RadiomicsGLCM
        glcm_features = glcm.RadiomicsGLCM(image_sitk, masked_image)
        print('4')
        glcm_features.enableFeatureByName('Autocorrelation')
        glcm_features.enableFeatureByName('Correlation')
        glcm_features.enableFeatureByName('JointEntropy')
        glcm_features.enableFeatureByName('JointEnergy')
        print('5')
        features =glcm_features.execute()


        a=features['Autocorrelation']
        c=features['Correlation']
        jointEntropy=features['JointEntropy']
        jointEnergy=features['JointEnergy']

        return a,c,jointEntropy,jointEnergy



    def params(self,image):
        otsu= self.detectar_zona_iluminada(image,125,len(image[0])//2,len(image)//2,len(image)//2)
        sinvasos = self.detect_blood_vessels(image)
        VasoArea=np.sum(image[sinvasos==255])/(len(image)*len(image[0]))
        ManchaArea=np.sum(image[otsu==255])/(len(image)*len(image[0]))
        #print(len(image)*len(image[0]))
        return VasoArea,ManchaArea
    

    def kmeans_segmentation(self, image, num_clusters):
        # Obtener los valores de píxeles de la imagen
        pixel_values = image.flatten().reshape(-1, 1)

        # Aplicar el algoritmo K-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(pixel_values)

        # Obtener las etiquetas de los píxeles
        labels = kmeans.labels_

        # Crear una imagen con los clusters
        segmented_image = np.zeros_like(image, dtype=np.uint8)
        for cluster in range(num_clusters):
            segmented_image[labels.reshape(image.shape) == cluster] = cluster * (255 // (num_clusters - 1))

        return segmented_image

    def detect_blood_vessels(self,gray):
        # Convertir la imagen a escala de grises
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #print("Shape de gray_image:", gray.shape)
        #print("Forma de la matriz gray:", gray.shape)
        #print("Tipo de datos de la matriz gray:", gray.dtype)
        # Asegurarse de que la matriz esté en el rango de 0 a 1
        gray_normalized = gray / np.max(gray)
        # Escalar la matriz a valores de píxeles en el rango de 0 a 255
        gray_scaled = (gray_normalized * 255).astype(np.uint8)
        #print("here")

        # Aplicar un filtro de realce de bordes
        edges = cv2.Canny(gray_scaled, 30, 70)
        #print("llegue aca")

        # Aplicar la transformada de Hough para detectar líneas
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=20, maxLineGap=5)
        #print("llegue aca 2")

        # Filtrar y dibujar las líneas correspondientes a los vasos sanguíneos
        vessels = np.zeros_like(gray)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > 30:
                    cv2.line(vessels, (x1, y1), (x2, y2), 255, 2)

        return vessels
    
    def detectar_zona_iluminada(self,image, threshold_value, centro_x, centro_y, radio):
        # Crear una máscara circular del mismo tamaño que la imagen
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, (centro_x, centro_y), radio, (255, 255, 255), -1)

        
        #print('hola')

        # Aplicar la máscara a la imagen en escala de grises
        masked_gray = cv2.bitwise_and(image, mask)
        # Aplicar una umbralización a la imagen en escala de grises
        _, threshold = cv2.threshold(masked_gray, threshold_value, 255, cv2.THRESH_BINARY)
        return threshold
    def deteccion_bordes_kirsch(self,imagen):
        # Convertir la imagen a escala de grises si es necesario
        if len(imagen.shape) > 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Aplicar el operador de Kirsch en todas las direcciones
        kernel_0 = np.array([[-3, -3, -3], [0, 0, 0], [3, 3, 3]], dtype=np.float32)
        kernel_45 = np.array([[-3, -3, 0], [-3, 0, 3], [0, 3, 3]], dtype=np.float32)
        kernel_90 = np.array([[-3, 0, 3], [-3, 0, 3], [-3, 0, 3]], dtype=np.float32)
        kernel_135 = np.array([[0, 3, 3], [-3, 0, 3], [-3, -3, 0]], dtype=np.float32)
        kernel_180 = np.array([[3, 3, 3], [0, 0, 0], [-3, -3, -3]], dtype=np.float32)
        kernel_225 = np.array([[3, 3, 0], [3, 0, -3], [0, -3, -3]], dtype=np.float32)
        kernel_270 = np.array([[3, 0, -3], [3, 0, -3], [3, 0, -3]], dtype=np.float32)
        kernel_315 = np.array([[0, -3, -3], [3, 0, -3], [3, 3, 0]], dtype=np.float32)

        bordes_0 = cv2.filter2D(imagen, -1, kernel_0)
        bordes_45 = cv2.filter2D(imagen, -1, kernel_45)
        bordes_90 = cv2.filter2D(imagen, -1, kernel_90)
        bordes_135 = cv2.filter2D(imagen, -1, kernel_135)
        bordes_180 = cv2.filter2D(imagen, -1, kernel_180)
        bordes_225 = cv2.filter2D(imagen, -1, kernel_225)
        bordes_270 = cv2.filter2D(imagen, -1, kernel_270)
        bordes_315 = cv2.filter2D(imagen, -1, kernel_315)

        # Obtener el valor absoluto de los bordes en todas las direcciones
        bordes_abs = np.abs(np.stack([bordes_0, bordes_45, bordes_90, bordes_135, bordes_180, bordes_225, bordes_270, bordes_315], axis=-1))

        # Obtener la magnitud de los bordes
        magnitud_bordes = np.max(bordes_abs, axis=-1)

        return magnitud_bordes
    
    def train_and_select_features(self,path, k=4):
        # Leer el archivo CSV y crear un DataFrame
        self.df = pd.read_csv(path)

        # Extraer la columna "Enfermedad" como variable de salida y eliminarla del DataFrame
        y = self.df.iloc[:, 0]  # La primera columna
        X = self.df.iloc[:, 1:]  # Resto de las columnas

        # Seleccionar las mejores K características utilizando f_classif como métrica
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)

        # Obtener los índices de las características seleccionadas
        selected_features = selector.get_support(indices=True)

        # Crear un DataFrame con las características seleccionadas
        X_new = pd.DataFrame(X_new, columns=X.columns[selected_features])

        # Dividir los datos en un conjunto de entrenamiento y un conjunto de prueba
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=0)

        # Crear y entrenar el clasificador (por ejemplo, K-Nearest Neighbors)
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, y_train)

        return clf, selected_features, X_train, X_test, y_train, y_test
    def classify_new_instance(self,instance, clf, selected_features):
        # Crear DataFrame con la nueva instancia y las características seleccionadas
        instance_df = pd.DataFrame([instance], columns=selected_features)

        # Realizar la predicción utilizando el modelo entrenado
        prediction = clf.predict(instance_df)

        return prediction
    

    def caracterizar_imagen(self):
        # Lógica para clasificar la imagen
        print("Botón Caracterizar presionado")
        #print("Shape de image:", self.loaded_image.shape)

        green_channel = self.loaded_image[:, :, 1]  # Obtener el canal verde
        #print("aca me ropo")
        #print("Shape de image:", green_channel.shape)

        #gray_image = color.rgb2gray(green_channel)  # Convertir a escala de grises
        #gray_image = cv2.cvtColor(green_channel, cv2.COLOR_GRAY2BGR)  # Convertir a escala de grises
        gray_image=green_channel
        #print("entro1")
        # Parámetros del círculo
        #print("Tipo de image:", type(gray_image))
        #print("Shape de image1:", gray_image.shape)


        center_x = len(gray_image[0])//2
        center_y = len(gray_image)//2
        radius = len(gray_image)//2
        #print('aca toy')
        img_sinfondo=self.eliminar_fondo_circular(gray_image,center_x,center_y,radius)
        #print('llegueeeeee')
        img_gamma=self.gamma_correction(img_sinfondo,1.7)

        a,c,entro,ener=self.cocu(img_gamma,center_x,center_y,radius)
        print("Autocorr:",a)
        print("Corr:",c)
        print("Entropia:",entro)
        print("Energia:",ener)
        print(float(a))

        Vasos, mancha =self.params(img_gamma)

        print('Area vasos',Vasos)
        print('Area mancha',mancha)
        
        self.vector_variables = [float(a), float(entro), float(ener),  mancha]
        print(self.vector_variables)
        table_data = [    ["Parámetro", "Valor"],     ["Autocorrelación", "{:.4f}".format(a)],    ["Correlación", "{:.4f}".format(c)],    ["Entropía", "{:.4f}".format(entro)],    ["Energía", "{:.4f}".format(ener)],    ["Área de vasos", "{:.4f}".format(Vasos)],    ["Área de mancha", "{:.4f}".format(mancha)]]


        sinvasos = self.detect_blood_vessels(gray_image)
        #print("entro2")
        tumor = self.kmeans_segmentation(gray_image, 3)
        #print("entro 3")
        i= self.detectar_zona_iluminada(self.loaded_image,125,len(gray_image[0])//2,len(gray_image)//2,len(gray_image)//2)
        #print("entro4")        
        kirsch=self.deteccion_bordes_kirsch(img_gamma)
        
        fig,ax=plt.subplots(2,2)
        f = 18
        #ORIGINAL
        im = ax[0,0].imshow(gray_image, vmin=0, vmax=255,cmap='gray')
        ax[0,0].set_title('Imagen original',fontsize=f)
        ax[0, 0].axis('off')  # Ocultar ejes
        
        #GAMMA
        im_vasos = ax[0,1].imshow(img_gamma, vmin=0, vmax=255,cmap='gray')
        ax[0,1].set_title('Corrección Gamma',fontsize=f)
        ax[0, 1].axis('off')  # Ocultar ejes
        
        #kirsch
        im_kmeans = ax[1,0].imshow(kirsch, vmin=0, vmax=255,cmap="gray")
        ax[1,0].set_title('Filtro de Kirsch',fontsize=f)
        ax[1, 0].axis('off')  # Ocultar ejes
        
        for i, ax in enumerate(ax.flatten()):
            if i == 3:  # Subplot 1,1 (derecha abajo)
                table = ax.table(cellText=table_data, loc='center', cellLoc='center')
                table.set_fontsize(12)
                table.scale(1.2, 1.2)
                ax.axis('off')
            else:
                ax.axis('off')

        # Ajustar el diseño de los subplots
        #im_mancha = ax[1,1].imshow(i, vmin=0, vmax=255,cmap="gray")
        #ax[1,1].set_title('Manchas',fontsize=f)
        #ax[1, 1].axis('off')  # Ocultar ejes

        plt.tight_layout()  # Ajustar los espacios entre los gráficos

        plt.savefig('grafs.png')  # Guardar la imagen temporalmente como archivo PNG
        plt.close()  # Cerrar la figura después de guardar la imagen
            

            # Crear objeto QImage a partir del archivo guardado
        qimage = QImage('grafs.png')
        pixmap = QPixmap.fromImage(qimage)

            # Acceder al QGraphicsView dentro del QStackedWidget
        stacked_widget = self.findChild(QStackedWidget, 'stackedWidget')

            # Obtener el índice de la pestaña "graf" en el QStackedWidget
        index = stacked_widget.indexOf(self.graf)

        # Mostrar la pestaña "graf"
        stacked_widget.setCurrentIndex(index)
        graphics_view = stacked_widget.widget(index).findChild(QtWidgets.QGraphicsView, 'graficos')

            # Mostrar la imagen en el QGraphicsView
        graphics_scene = QtWidgets.QGraphicsScene()
        graphics_scene.addPixmap(pixmap)
        graphics_view.setScene(graphics_scene)
        graphics_view.fitInView(graphics_scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
    def diagnosticar_imagen(self):
            # Lógica para mejorar la imagen
            print("Botón Diagnostico presionado")
            # Ruta hacia el archivo CSV con los datos de entrenamiento
            ruta_archivo = 'C:/Users/Luciana/Downloads/BigTablenueva2.csv'
            # Entrenar el modelo y seleccionar características
            modelo, variables_seleccionadas, X_train, X_test, y_train, y_test = self.train_and_select_features(ruta_archivo)
           
            print(variables_seleccionadas)
            # Realizar la clasificación de la nueva imagen
            prediccion = self.classify_new_instance(self.vector_variables, modelo, variables_seleccionadas)
            df=self.df
            # Imprimir el diagnóstico
            print("Diagnóstico:", prediccion)
            #RECIBE EL VECTOR DE CARACTERISTICAS DE LA IMAGEN A BUSCAR
            diagnosticos = {
                0: 'Sin presencia de anomalías',
                1: 'Neovascularización Coroidea',
                2: 'Retinopatía Diabética'
            }
            diag = int(prediccion)
            print(diag)
            self.texto_etiqueta = "Diagnóstico sugerido mediante algoritmo: %s" % diagnosticos.get(diag)
            print(self.texto_etiqueta)
            self.diagnostico.setText(self.texto_etiqueta)
            colors = ['orange', 'maroon', 'navy']
            labels = ['Normal', 'CNV', 'Diabetes']
            markers = ['s', 'X', 'P']
            #punto=[1.042764,1.321693e-01,0.968226,2.915568]
            # Definir los pares de etiquetas para los ejes
            axis_labels = [('Manchas','JointEntropy'), ('Autocorrelation', 'JointEntropy'), ('JointEntropy', 'JointEnergy'), ('Manchas', 'Autocorrelation')]
            punto_data={
                'Autocorrelation':self.vector_variables[0],
                'JointEntropy':self.vector_variables[1],
                'JointEnergy':self.vector_variables[2], 
                'Manchas':self.vector_variables[3]
            }
            try:
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
                for i, ax in enumerate(axes.flatten()):
                    
                    x_label, y_label = axis_labels[i]
                    print(x_label, y_label)
                    for j, label in enumerate(labels):
                        ax.scatter(df.loc[df['Enfermedad'] == j, x_label], df.loc[df['Enfermedad'] == j, y_label], c=colors[j], marker=markers[j], label=label)
                    ax.scatter(punto_data[x_label], punto_data[y_label], c='black', marker='o',s=80, label='Muestra')
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.legend()

                # Ajustar el espaciado entre los subplots
                plt.tight_layout()
            except Exception as e:
                print("Error:", e)
            
            plt.savefig('diag.png')  # Guardar la imagen temporalmente como archivo PNG
            
            plt.close()  # Cerrar la figura después de guardar la imagen
                

            # Crear objeto QImage a partir del archivo guardado
            qimage = QImage('diag.png')
            pixmap = QPixmap.fromImage(qimage)

        # Acceder al QGraphicsView dentro del QStackedWidget
            stacked_widget = self.findChild(QStackedWidget, 'stackedWidget')

            # Obtener el índice de la pestaña "im" en el QStackedWidget
            index = stacked_widget.indexOf(self.Diag)

            # Mostrar la pestaña "im"
            stacked_widget.setCurrentIndex(index)
            graphics_view = stacked_widget.widget(index).findChild(QtWidgets.QGraphicsView, 'grafDiag')
            graphics_scene = QtWidgets.QGraphicsScene()
            graphics_scene.addPixmap(pixmap)
            graphics_view.setScene(graphics_scene)
            graphics_view.fitInView(graphics_scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

           
    
    def KNN_GRAF(self,punto,df):
        #RECIBE EL VECTOR DE CARACTERISTICAS DE LA IMAGEN A BUSCAR
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        colors = ['orange', 'maroon', 'navy']
        labels = ['Normal', 'CNV', 'Diabetes']
        markers = ['s', 'X', 'P']
        #punto=[1.042764,1.321693e-01,0.968226,2.915568]
        # Definir los pares de etiquetas para los ejes
        axis_labels = [('Manchas','JointEntropy'), ('Autocorrelation', 'JointEntropy'), ('JointEntropy', 'JointEnergy'), ('Manchas', 'Autocorrelation')]
        punto_data={
            'Autocorrelation':punto[0],
            'JointEntropy':punto[1],
            'JointEnergy':punto[2], 
            'Manchas':punto[3]
        }
        for i, ax in enumerate(axes.flatten()):
            x_label, y_label = axis_labels[i]
            for j, label in enumerate(labels):
                ax.scatter(df.loc[df['Enfermedad'] == j, x_label], df.loc[df['Enfermedad'] == j, y_label], c=colors[j], marker=markers[j], label=label)
            ax.scatter(punto_data[x_label], punto_data[y_label], c='black', marker='o',markersize=3, label='Muestra')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.legend()

        # Ajustar el espaciado entre los subplots
        plt.tight_layout()
        plt.savefig('diag.png')  # Guardar la imagen temporalmente como archivo PNG
        plt.close()  # Cerrar la figura después de guardar la imagen
            

        # Crear objeto QImage a partir del archivo guardado
        qimage = QImage('diag.png')
        pixmap = QPixmap.fromImage(qimage)

       # Acceder al QGraphicsView dentro del QStackedWidget
        stacked_widget = self.findChild(QStackedWidget, 'stackedWidget')

        # Obtener el índice de la pestaña "im" en el QStackedWidget
        index = stacked_widget.indexOf(self.Diag)

        # Mostrar la pestaña "im"
        stacked_widget.setCurrentIndex(index)
        graphics_view = stacked_widget.widget(index).findChild(QtWidgets.QGraphicsView, 'imagenes')
        graphics_scene = QtWidgets.QGraphicsScene()
        graphics_scene.addPixmap(pixmap)
        graphics_view.setScene(graphics_scene)
        graphics_view.fitInView(graphics_scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        #plt.show()
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    interfaz = InterfazPib()
    interfaz.show()
    sys.exit(app.exec_())

